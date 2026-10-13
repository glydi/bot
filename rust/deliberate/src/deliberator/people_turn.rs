//! The people-first turn: UNDERSTAND (rules, else the small model with a
//! JSON schema), DECIDE + ANSWER in code from the graph and the visit
//! notebook, the model only for chit-chat, the hand-off for anything out
//! of zone. Every turn lands in the notebook and the learning log.

#![allow(
    clippy::single_match_else,
    clippy::format_push_string,
    clippy::map_unwrap_or
)]

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use common::{Command, EntityId, Payload, Priority};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{SET_NAME_KIND, SET_NAME_TARGET, Session, TurnEnd, barge_in};
use crate::ChatEvent;
use crate::backend::{ChatRequest, LlmError};
use crate::converse::{self, PERSONA, pick};
use crate::handoff::Handoff;
use crate::people::{FactOp, Relation};
use crate::prompt::Message;
use crate::understand::{self, Mention, UnderstandModel, Understanding};
use crate::visit::Awaiting;
use common::Observation;

/// The people-first pipeline's parts. `None` in [`super::Config::people`]
/// keeps the older lanes.
#[derive(Clone)]
pub struct People {
    /// The small model that understands; `None` = rules only.
    pub understander: Option<Arc<dyn UnderstandModel>>,
    /// Where out-of-zone questions go.
    pub handoff: Arc<dyn Handoff>,
    /// Say a short "Mm," when a model reply is slow to start
    /// (`GLYDI_THINK_ACK=1`). Off by default: in `simulate.py` the
    /// acknowledgement became the whole first line and hid the reply.
    pub think_ack: bool,
    /// The stronger model for the slow path (questions that need
    /// thinking); `None` uses the session's own backend.
    pub deep: Option<Arc<dyn crate::backend::ChatBackend>>,
    /// Told after every turn whether the bot now waits for a name (its
    /// own, a relation's or a companion's), so the hearing side may read a
    /// bare one- or two-word reply as a name (`NameFixer::expect_name`).
    pub expecting_name: Option<ExpectName>,
}

/// See [`People::expecting_name`].
pub type ExpectName = Arc<dyn Fn(bool) + Send + Sync>;

impl std::fmt::Debug for People {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("People")
            .field("understander", &self.understander.is_some())
            .field("deep", &self.deep.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for People {
    fn default() -> Self {
        Self {
            understander: None,
            handoff: Arc::new(crate::handoff::DefaultHandoff),
            think_ack: false,
            deep: None,
            expecting_name: None,
        }
    }
}

/// Clarifying lines, rotated.
pub const CLARIFY: &[&str] = &[
    "Sorry, could you say that again?",
    "I didn't quite get that. Once more?",
    "Sorry, say that again for me?",
];
/// Asked of a stranger once per visit.
pub const ASK_NAME: &str = "Hello! I'm Glydi. What's your name?";
/// Asked when an unknown face stands beside someone known.
pub const WHO_FRIEND: &str = "Who's your friend?";
/// Chit-chat reply ceiling in words.
pub const CHAT_WORDS: usize = 24;

/// Intents whose decided line is said in the model's own words.
const REPHRASE: &[&str] = &[
    "greet",
    "bye",
    "how_are_you",
    "ask_about_bot",
    "ask_about_me",
    "ask_about_person",
    "tell_fact",
    "tell_relation",
    "tell_name",
    "correct_name",
    "forget_me",
    "everyday",
    "out_of_zone",
    "unclear",
];

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

fn bare_name(text: &str) -> Option<String> {
    let w = understand::words(text);
    let w: Vec<&str> = w
        .iter()
        .map(String::as_str)
        .filter(|x| {
            !matches!(
                *x,
                "it's" | "its" | "is" | "her" | "his" | "name" | "she's" | "he's"
            )
        })
        .collect();
    (w.len() == 1 && crate::voice::plausible_name(w[0]) && w[0].len() >= 2).then(|| {
        let mut c = w[0].chars();
        c.next()
            .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
            .unwrap_or_default()
    })
}

impl Session {
    /// Whether this session runs the people-first pipeline.
    pub fn people_first(&self) -> bool {
        self.people.is_some()
    }

    /// The known speaker (not a track) and their name.
    fn who(&self, speaker: Option<&EntityId>) -> (Option<EntityId>, Option<String>) {
        let id = speaker
            .filter(|s| !s.is_track())
            .cloned()
            .or_else(|| self.current_person.as_ref().map(|(i, _)| i.clone()));
        let name = id.as_ref().and_then(|i| {
            self.current_person
                .as_ref()
                .filter(|(c, _)| c == i)
                .map(|(_, n)| n.clone())
                .or_else(|| self.facts.name_of(i))
                .or_else(|| self.name_of(Some(i)))
        });
        (id, name)
    }

    /// Known people in view, and an unknown face that is not the speaker.
    fn room_people(&self, speaker: Option<&EntityId>) -> (Vec<EntityId>, Option<EntityId>) {
        let view = (self.snapshot)();
        let focus = view.working.focus.clone();
        let known = view
            .people
            .iter()
            .filter(|p| p.is_known())
            .map(|p| p.id.clone())
            .collect();
        let stranger = view
            .people
            .iter()
            .filter(|p| p.id.is_track() && !p.is_known())
            .map(|p| p.id.clone())
            .find(|id| Some(id) != speaker && Some(id) != focus.as_ref());
        (known, stranger)
    }

    /// The greeting line for a known arrival, from the graph (people
    /// mode). Adds "Who's your friend?" when an unknown face is beside.
    pub(super) fn people_greeting(&mut self, id: &EntityId, name: &str) -> String {
        let (present, stranger) = self.room_people(Some(id));
        // A plain hello by default: what the bot knows about them (family
        // here today, last visit) is said only when they ask.
        // GLYDI_GREET_FACTS=1 brings back the remembering greeting.
        let mut line = if std::env::var("GLYDI_GREET_FACTS").is_ok_and(|v| v.trim() == "1") {
            converse::greeting(id, name, self.facts.as_ref(), unix_now(), &present)
        } else {
            let since = self
                .facts
                .previous_visit(id)
                .map(|t| unix_now() - t)
                .and_then(since_phrase);
            match since {
                Some(p) => format!("{}, {name}! {p}", time_greeting()),
                None => format!("{}, {name}!", time_greeting()),
            }
        };
        if stranger.is_some() {
            line.push(' ');
            line.push_str(WHO_FRIEND);
            self.visit.awaiting = Some(Awaiting::Companion);
        }
        self.visit.greeted = true;
        self.sync_expect_name();
        line
    }

    /// One people-first turn.
    #[allow(clippy::too_many_lines)]
    pub(super) async fn people_turn(
        &mut self,
        text: &str,
        speaker: Option<&EntityId>,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: &CancellationToken,
    ) -> Result<TurnEnd, LlmError> {
        let started = self.clock.now();
        let said_before = self.said.all().len();
        self.first_say.set(None);
        if !self.typed && self.said.echoes(text) {
            tracing::info!(text, "self-echo dropped");
            return Ok(TurnEnd::Done);
        }
        let now_i = self.clock.now();
        if self
            .attention_at
            .is_some_and(|t| now_i.saturating_duration_since(t) >= super::VISIT_GAP)
        {
            self.new_visit();
        }
        let emergency = crate::templates::is_emergency(text);
        // A noisy foyer: with a camera, speech is for us when someone is
        // in focus, a known voice speaks, or it names or greets us.
        if self.attention && !self.typed && !emergency {
            let view = (self.snapshot)();
            let seeing = view.self_model().can_see(view.at);
            let for_me = if seeing {
                view.working.focus.is_some()
                    || speaker.is_some()
                    || self.visit.awaiting.is_some()
                    || crate::voice::calls_me(text)
                    || understand::first_name_in(text).is_some()
            } else {
                true
            };
            if !for_me {
                tracing::info!(text, "not addressed to me; ignored");
                return Ok(TurnEnd::Done);
            }
        }
        self.attention_at = Some(now_i);
        if let Some(l) = &mut self.learn {
            l.heard(text, now_i);
        }
        // Someone else in front now: that visit is over.
        if let Some(id) = speaker.filter(|s| !s.is_track())
            && self.visit.key.as_ref().is_some_and(|k| k != id)
        {
            self.new_visit();
        }
        self.ui("thinking");
        // The safety net: one calm fixed line, before anything else.
        if let Some(line) = crate::templates::emergency_line(text) {
            self.speak_line(line.to_owned());
            self.turn_meta = Some(("emergency".into(), 1.0, "{\"by\":\"rules\"}".into()));
            self.learn_turn(
                text,
                speaker,
                crate::learn::Lane::People,
                started,
                said_before,
                None,
            );
            self.ui("idle");
            return Ok(TurnEnd::Done);
        }
        let (mut me, mut my_name) = self.who(speaker);
        if self.visit.name.is_none() {
            self.visit.name.clone_from(&my_name);
        }
        // An answer the bot was waiting for, by rule.
        let awaited = match (&self.visit.awaiting, bare_name(text)) {
            (Some(Awaiting::Name), Some(n)) => {
                let mut u = Understanding::of("tell_name");
                u.people.push(Mention {
                    name: n,
                    ..Mention::default()
                });
                Some(u)
            }
            (Some(Awaiting::Companion), Some(n)) => {
                let mut u = Understanding::of("tell_relation");
                u.people.push(Mention {
                    name: n,
                    relation_to_speaker: Some("friend".into()),
                    reference: Some("your friend".into()),
                });
                Some(u)
            }
            (Some(Awaiting::RelationName(w)), Some(n)) => {
                let mut u = Understanding::of("tell_relation");
                u.people.push(Mention {
                    name: n,
                    relation_to_speaker: Some(w.clone()),
                    reference: Some(format!("my {w}")),
                });
                Some(u)
            }
            _ => None,
        };
        let understood = match awaited {
            Some(u) => understand::Understood {
                u,
                by: "rules",
                valid: true,
                retries: 0,
                ms: 0,
            },
            None => {
                let notebook = converse::layered(
                    self.facts.as_ref(),
                    me.as_ref(),
                    my_name.as_deref(),
                    &self.visit,
                );
                let model = self.people.as_ref().and_then(|p| p.understander.clone());
                understand::understand(model.as_deref(), &notebook, text).await
            }
        };
        tracing::info!(
            intent = understood.u.intent,
            confidence = understood.u.confidence,
            by = understood.by,
            valid = understood.valid,
            retries = understood.retries,
            ms = understood.ms,
            "understood"
        );
        self.visit.awaiting = None;
        let u = understood.u.clone();
        let mut intent = u.intent.as_str();
        // "Diya started dancing classes": a person we know of is not out of
        // zone, whatever the model says.
        if intent == "out_of_zone" {
            let lower = text.to_lowercase();
            let edges = me.as_ref().map(|m| self.facts.edges(m)).unwrap_or_default();
            if edges
                .iter()
                .any(|e| lower.contains(&e.other_name.to_lowercase()))
                || self
                    .visit
                    .refs
                    .iter()
                    .any(|(_, n)| lower.contains(&n.to_lowercase()))
            {
                intent = "tell_fact";
            }
        }
        // The model's out_of_zone and everyday are checked against the
        // words: a statement ("the printer broke again", "we just moved to
        // Kondapur") is never handed off or answered with the date.
        if understood.by != "rules" {
            let joined = understand::words(text).join(" ");
            let statement = || {
                if u.facts.is_empty() {
                    "chit_chat"
                } else {
                    "tell_fact"
                }
            };
            if intent == "out_of_zone" && !understand::asks(text) {
                intent = statement();
            } else if intent == "everyday" && !understand::everyday(&joined) {
                intent = if understand::asks(text) {
                    "out_of_zone"
                } else {
                    statement()
                };
            }
        }
        // Two speeds: a question that needs thinking (advice, planning,
        // reasoning over what they told before) goes to the slow path; an
        // open question the understander is unsure of goes there too,
        // rather than a guess or "say that again".
        let unsure_question = understood.by == "model"
            && u.heard_ok
            && matches!(intent, "chit_chat" | "unclear" | "feeling")
            && u.confidence < DEEP_BELOW
            && understand::open_question(text);
        let rule_deep = matches!(intent, "chit_chat" | "unclear" | "feeling" | "tell_fact")
            && understand::deep_question(text);
        let labelled = intent == "deep";
        if unsure_question || rule_deep {
            intent = "deep";
        }
        // The fallback (no model answer) is small talk on purpose: a
        // friendly line, not "say that again" to every sentence.
        if intent != "deep"
            && understood.by != "fallback"
            && (!u.heard_ok || u.confidence < understand::MIN_CONFIDENCE)
        {
            intent = "unclear";
        }
        tracing::info!(
            route = if intent == "deep" { "slow" } else { "fast" },
            why = if intent != "deep" {
                "intent"
            } else if labelled {
                "labelled deep"
            } else if rule_deep {
                "question shape"
            } else {
                "unsure question"
            },
            intent,
            confidence = u.confidence,
            "router"
        );
        let salt = self.visit.exchanges.len() + text.len();
        let last = self.said.all().last().copied();
        let now = unix_now();
        let line: Option<String> = match intent {
            "greet" => Some(match (&me, &my_name) {
                (Some(id), Some(n)) if !self.visit.greeted => {
                    let id = id.clone();
                    let n = n.clone();
                    self.people_greeting(&id, &n)
                }
                (Some(_), Some(n)) => {
                    pick(&["Hello again, {n}!", "Hi {n}!"], salt, last).replace("{n}", n)
                }
                _ if self.visit_asked_name => "Hello again!".to_owned(),
                _ => {
                    self.visit_asked_name = true;
                    self.visit.awaiting = Some(Awaiting::Name);
                    ASK_NAME.to_owned()
                }
            }),
            "bye" => Some(match &my_name {
                Some(n) => pick(
                    &["Bye {n}! See you soon.", "Goodbye {n}, take care!"],
                    salt,
                    last,
                )
                .replace("{n}", n),
                None => "Bye! See you soon.".to_owned(),
            }),
            "tell_name" | "correct_name" => {
                let name = u
                    .people
                    .iter()
                    .find(|p| p.relation_to_speaker.is_none() && !p.name.is_empty())
                    .map(|p| p.name.clone())
                    .or_else(|| understand::first_name_in(text))
                    .or_else(|| {
                        (intent == "correct_name")
                            .then(|| said_name(text, my_name.as_deref()))
                            .flatten()
                    });
                match name {
                    Some(n) => {
                        let known_before = self.facts.resolve_name(&n).is_some()
                            && intent == "tell_name"
                            && me.is_none();
                        let view = (self.snapshot)();
                        self.enrol_name(&n, &view);
                        drop(view);
                        me = self.current_person.as_ref().map(|(i, _)| i.clone());
                        my_name = Some(n.clone());
                        self.visit.name = Some(n.clone());
                        self.visit.key.clone_from(&me);
                        let mut rel = self.store_relations(me.as_ref(), &u, text);
                        let pending = std::mem::take(&mut self.visit.pending_people);
                        if !pending.is_empty() {
                            let mut pu = Understanding::of("tell_relation");
                            pu.people = pending;
                            rel = rel.or_else(|| self.store_relations(me.as_ref(), &pu, text));
                        }
                        if let Some(id) = &me {
                            for f in &u.facts {
                                converse::write_fact(self.facts.as_ref(), id, f, false);
                            }
                        }
                        let mut l = if known_before && let Some(id) = me.clone() {
                            // Back again: the people moment, from the graph.
                            self.people_greeting(&id, &n)
                        } else if intent == "correct_name" {
                            format!("Sorry about that! {n} it is.")
                        } else {
                            format!("Nice to meet you, {n}!")
                        };
                        if let Some(r) = rel {
                            l.push(' ');
                            l.push_str(&r);
                        }
                        Some(l)
                    }
                    None => {
                        self.visit.awaiting = Some(Awaiting::Name);
                        Some("Sorry! What's your name?".to_owned())
                    }
                }
            }
            "tell_relation" => {
                let unnamed = u
                    .people
                    .iter()
                    .find(|p| p.name.is_empty() && p.relation_to_speaker.is_some());
                if let Some(m) = unnamed {
                    let w = m.relation_to_speaker.clone().unwrap_or_default();
                    self.visit.awaiting = Some(Awaiting::RelationName(w.clone()));
                    Some(format!("Hello! What's your {w}'s name?"))
                } else if me.is_none() {
                    self.visit.awaiting = Some(Awaiting::Name);
                    self.visit_asked_name = true;
                    let named: Vec<Mention> = u
                        .people
                        .iter()
                        .filter(|p| !p.name.is_empty() && p.relation_to_speaker.is_some())
                        .cloned()
                        .collect();
                    let line = match named.first() {
                        Some(m) => match m.reference.as_deref() {
                            Some(r) if r.contains("'s ") => {
                                format!("Lovely, {r}! And what's your name?")
                            }
                            _ => format!("Hi {}! And what's your name?", m.name),
                        },
                        None => "Nice! And what's your name?".to_owned(),
                    };
                    self.visit.pending_people.extend(named);
                    Some(line)
                } else {
                    let companion = self.room_people(speaker).1;
                    let first = u.people.first().map(|p| p.name.clone()).unwrap_or_default();
                    let beside = companion.is_some();
                    if let (Some(track), false) = (companion, first.is_empty()) {
                        self.name_companion(&track, &first);
                    }
                    let r = self.store_relations(me.as_ref(), &u, text);
                    if let Some(id) = &me {
                        for f in &u.facts {
                            converse::write_fact(self.facts.as_ref(), id, f, false);
                        }
                    }
                    Some(match (r, first.is_empty()) {
                        (_, false) if beside => format!("Hi {first}! Nice to meet you."),
                        (Some(r), _) => r,
                        (None, false) => format!("Hi {first}! Nice to meet you."),
                        (None, true) => "Nice!".to_owned(),
                    })
                }
            }
            "tell_fact" => {
                let retract = converse::retracts(text);
                self.store_relations(me.as_ref(), &u, text);
                let mut ops = Vec::new();
                if let Some(id) = &me {
                    let replaces = converse::replaces(text);
                    for f in &u.facts {
                        ops.push(converse::write_fact_replacing(
                            self.facts.as_ref(),
                            id,
                            f,
                            retract,
                            replaces,
                        ));
                    }
                }
                Some(if ops.iter().any(|o| matches!(o, FactOp::Delete(_))) {
                    "Okay, I've forgotten that.".to_owned()
                } else if ops.iter().any(|o| matches!(o, FactOp::Update(_))) {
                    "Got it, I've updated that.".to_owned()
                } else if !ops.is_empty() && ops.iter().all(|o| *o == FactOp::Noop) {
                    "Yes, you told me that before!".to_owned()
                } else {
                    pick(
                        &["Good to know!", "I'll remember that.", "Oh, nice!"],
                        salt,
                        last,
                    )
                })
            }
            "ask_about_person" => {
                let target = u
                    .people
                    .first()
                    .and_then(|m| {
                        converse::resolve(
                            m,
                            u.refers_to.as_deref(),
                            me.as_ref(),
                            self.facts.as_ref(),
                            &self.visit,
                        )
                    })
                    .or_else(|| {
                        u.refers_to.as_deref().and_then(|r| {
                            let m = Mention {
                                reference: Some(r.to_owned()),
                                ..Mention::default()
                            };
                            converse::resolve(
                                &m,
                                None,
                                me.as_ref(),
                                self.facts.as_ref(),
                                &self.visit,
                            )
                        })
                    });
                Some(match target {
                    Some(t) => {
                        let present = self.room_people(speaker).0;
                        let who = text.to_lowercase().starts_with("who");
                        let l = converse::about_person(
                            &t,
                            me.as_ref(),
                            self.facts.as_ref(),
                            now,
                            &present,
                            who,
                        );
                        self.visit.refer(t.0, &t.1);
                        l
                    }
                    None => match u.people.first() {
                        Some(m) if m.relation_to_speaker.is_some() && m.name.is_empty() => {
                            let w = m.relation_to_speaker.clone().unwrap_or_default();
                            self.visit.awaiting = Some(Awaiting::RelationName(w.clone()));
                            format!("I don't know your {w} yet. What's their name?")
                        }
                        Some(m) if !m.name.is_empty() => {
                            format!("I don't know {} yet.", m.name)
                        }
                        _ => "Who do you mean?".to_owned(),
                    },
                })
            }
            "how_are_you" => Some(match &my_name {
                Some(n) => format!("I'm doing great, thanks {n}! How about you?"),
                None => "I'm doing great, thanks! How about you?".to_owned(),
            }),
            "ask_about_bot" => Some(
                "I'm Glydi, the robot at the door. I remember people, and who they come with."
                    .to_owned(),
            ),
            "ask_about_me" => Some(match (&me, &my_name) {
                // A specific question ("what is my favourite colour?"):
                // answer just that, from the fact that matches it best.
                (Some(id), Some(_)) if asked_detail(text).is_some() => {
                    let words = asked_detail(text).unwrap_or_default();
                    let facts: Vec<String> = self
                        .facts
                        .recall(id)
                        .into_iter()
                        .filter(|f| converse::shown(f))
                        .collect();
                    let best = facts
                        .iter()
                        .map(|f| {
                            let lf = f.to_ascii_lowercase();
                            (words.iter().filter(|w| lf.contains(w.as_str())).count(), f)
                        })
                        .filter(|(n, _)| *n > 0)
                        .max_by_key(|(n, _)| *n);
                    match best {
                        Some((_, f)) => {
                            let said = second_person(f);
                            if said.starts_with("your ") {
                                format!("You told me {said}.")
                            } else {
                                format!("You told me you {said}.")
                            }
                        }
                        None => "You haven't told me that yet.".to_owned(),
                    }
                }
                (Some(_), Some(n)) if asks_name_only(text) => format!("You're {n}."),
                (Some(id), Some(n)) => {
                    let edges = self.facts.edges(id);
                    let mut l = format!("You're {n}");
                    if let Some(e) = edges.first() {
                        l.push_str(&format!(
                            ", {}'s {}",
                            e.other_name,
                            converse::relation_noun(&e.relation)
                        ));
                    }
                    l.push('.');
                    let facts: Vec<String> = self
                        .facts
                        .recall(id)
                        .into_iter()
                        .filter(|f| converse::shown(f))
                        .collect();
                    if let Some(f) = facts.last() {
                        let said = second_person(f);
                        if said.starts_with("your ") {
                            l.push_str(&format!(" You told me {said}."));
                        } else {
                            l.push_str(&format!(" You told me you {said}."));
                        }
                    }
                    l
                }
                _ => {
                    self.visit.awaiting = Some(Awaiting::Name);
                    self.visit_asked_name = true;
                    "I don't know your name yet. What is it?".to_owned()
                }
            }),
            "forget_me" => {
                if let Some(id) = &me {
                    self.facts.forget(id);
                }
                self.forget_person_here();
                me = None;
                my_name = None;
                Some("Okay, I've forgotten you. Bye for now!".to_owned())
            }
            "everyday" => Some(everyday_answer(text)),
            "deep" => match self
                .deep_line(text, me.as_ref(), my_name.as_deref(), obs, cancel)
                .await
            {
                Ok(l) => Some(l),
                Err(()) => {
                    self.ui("idle");
                    return Ok(TurnEnd::Cancelled);
                }
            },
            "out_of_zone" => {
                let h = self
                    .people
                    .as_ref()
                    .map(|p| Arc::clone(&p.handoff))
                    .unwrap_or_else(|| Arc::new(crate::handoff::DefaultHandoff));
                Some(h.answer(text, my_name.as_deref()).await)
            }
            "chit_chat" | "feeling" => {
                self.store_relations(me.as_ref(), &u, text);
                if let Some(id) = &me {
                    for f in &u.facts {
                        converse::write_fact(self.facts.as_ref(), id, f, false);
                    }
                }
                match self
                    .chat_line(text, me.as_ref(), my_name.as_deref(), obs, cancel)
                    .await
                {
                    Ok(Some(l)) => Some(l),
                    Ok(None) => None,
                    Err(()) => {
                        self.ui("idle");
                        return Ok(TurnEnd::Cancelled);
                    }
                }
            }
            _ => Some(pick(CLARIFY, salt, last)),
        };
        tracing::info!(intent, by = understood.by, "people answered");
        // Code decided WHAT to say; the model says it in its own words so
        // the bot doesn't sound canned. Exact lines (name confirmation,
        // emergency, privacy, time/sums, the hand-off) stay as they are.
        let line = match line {
            Some(l)
                if !cfg!(test)
                    && std::env::var("GLYDI_REPHRASE").map_or(true, |v| v.trim() != "0")
                    && REPHRASE.contains(&intent)
                    && !l.ends_with("Did I get that right?")
                    && !fixed_line(intent, &l) =>
            {
                match self
                    .rephrase(text, &l, my_name.as_deref(), obs, cancel)
                    .await
                {
                    Ok(r) => Some(r),
                    Err(()) => {
                        self.ui("idle");
                        return Ok(TurnEnd::Cancelled);
                    }
                }
            }
            other => other,
        };
        if let Some(l) = line {
            if std::mem::take(&mut self.streamed) {
                // Already said while the model wrote it.
                self.attention_at = Some(self.clock.now());
                tracing::info!(line = l, "proactive");
                self.conversation.push(Message::assistant(&l));
                self.said.push(&l);
            } else {
                self.speak_line(l);
            }
        }
        if intent == "bye" {
            // The visit ends; what it leaves is summarised now.
            self.visit.key.clone_from(&me);
            self.new_visit();
            self.current_person = None;
            self.turn_meta = Some((u.intent.clone(), u.confidence, understood.log_json()));
            self.learn_turn(
                text,
                me.as_ref(),
                crate::learn::Lane::People,
                started,
                said_before,
                None,
            );
            self.sync_expect_name();
            self.ui("idle");
            return Ok(TurnEnd::Done);
        }
        // The visit notebook, after every turn.
        self.visit.key.clone_from(&me);
        self.visit.name.clone_from(&my_name);
        self.visit.intent = Some((u.intent.clone(), u.confidence));
        for m in &u.people {
            if !m.name.is_empty()
                && let Some(id) = self.facts.resolve_name(&m.name)
            {
                self.visit.refer(id, &m.name);
            }
        }
        self.turn_meta = Some((u.intent.clone(), u.confidence, understood.log_json()));
        self.learn_turn(
            text,
            me.as_ref(),
            crate::learn::Lane::People,
            started,
            said_before,
            None,
        );
        self.sync_expect_name();
        self.ui("idle");
        Ok(TurnEnd::Done)
    }

    /// Tell the hearing side whether a name is expected now.
    fn sync_expect_name(&self) {
        if let Some(hook) = self.people.as_ref().and_then(|p| p.expecting_name.as_ref()) {
            hook(matches!(
                self.visit.awaiting,
                Some(Awaiting::Name | Awaiting::Companion | Awaiting::RelationName(_))
            ));
        }
    }

    /// Store every relation the utterance gave; the confirmation line for
    /// the first, if any.
    fn store_relations(
        &self,
        me: Option<&EntityId>,
        u: &Understanding,
        text: &str,
    ) -> Option<String> {
        let me = me?;
        let mut confirm = None;
        for m in &u.people {
            let (Some(word), false) = (m.relation_to_speaker.as_deref(), m.name.is_empty()) else {
                continue;
            };
            let Some(r) = Relation::from_word(word) else {
                continue;
            };
            // `m` is `r` of me, so I am `r.inverse()` of `m`.
            let mine = r.inverse();
            self.facts
                .relate_people(me, &mine, &m.name, text, u.confidence.max(0.5));
            // The word itself ("son", not just "child"), on them.
            if let (Some(other), Some(my)) =
                (self.facts.resolve_name(&m.name), self.facts.name_of(me))
            {
                let f = format!("{}{my}", converse::relation_word_fact(word));
                if !self.facts.recall(&other).contains(&f) {
                    self.facts.remember(&other, &f);
                }
            }
            if confirm.is_none() {
                confirm = Some(match m.reference.as_deref() {
                    Some(r) if r.contains("'s ") => format!("I'll remember you're {r}."),
                    _ => format!("I'll remember {} is your {word}.", m.name),
                });
            }
        }
        confirm
    }

    /// Bind `name` to the unknown face beside the speaker.
    fn name_companion(&mut self, track: &EntityId, name: &str) {
        match self.facts.remember_name(Some(track), name) {
            Ok(entity) => {
                tracing::info!(%track, name, %entity, "companion named");
                let mut payload = serde_json::json!({ "entity": entity.as_str(), "name": name });
                if let Some(t) = track
                    .as_str()
                    .strip_prefix("track:")
                    .and_then(|n| n.parse::<u32>().ok())
                {
                    payload["track"] = serde_json::json!(t);
                }
                self.commands.push(
                    Command::new(SET_NAME_TARGET, SET_NAME_KIND, Priority::Deliberate)
                        .with_payload(Payload::Text(payload.to_string())),
                );
            }
            Err(e) => tracing::info!(error = e, "companion not named"),
        }
    }

    /// Chit-chat: the model writes one short warm line from the layered
    /// prompt. `Err` = cancelled.
    /// Say `content` in the bot's own words, keeping every name and number
    /// in it. Falls back to `content` itself when the model is slow, leaks
    /// or drops something that mattered.
    async fn rephrase(
        &mut self,
        text: &str,
        content: &str,
        name: Option<&str>,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: &CancellationToken,
    ) -> Result<String, ()> {
        let mut m = vec![Message::system(format!(
            "{PERSONA}
{}",
            now_line()
        ))];
        for e in self.visit.recent() {
            m.push(Message::user(e.heard.clone()));
            m.push(Message::assistant(e.said.clone()));
        }
        let who = name.map(|n| format!(" (this is {n})")).unwrap_or_default();
        m.push(Message::user(format!(
            "[they said]{who} {text}
[say this, in your own warm words, keeping every name and fact and its meaning; one or two short sentences; add no new question or fact] {content}"
        )));
        let req = ChatRequest {
            messages: m,
            tools: Vec::new(),
            max_tokens: 50,
            temperature: 0.8,
            json_object: false,
        };
        let stream = self.backend.chat(req);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        let need = required_words(content);
        let carries = |s: &str| {
            let low = s.to_lowercase();
            need.iter().all(|w| low.contains(w.as_str()))
                && (!content.trim_end().ends_with('?') || s.contains('?'))
                && (content.contains('?') || !s.contains('?'))
        };
        let (raw, first) = self
            .stream_say_first(stream, deadline, obs, cancel, &carries)
            .await?;
        if let Some(first) = first {
            return Ok(self.finish_streamed(first, raw.as_deref(), &need));
        }
        let Some(raw) = raw else {
            return Ok(content.to_owned());
        };
        let Some(l) = crate::voice::speakable(&raw, CHAT_WORDS + 6) else {
            tracing::info!(raw, content, "rephrase unsayable; plain line");
            return Ok(content.to_owned());
        };
        let real = self.real_context();
        let low = l.to_lowercase();
        // Every name and number the content carried must survive.
        // Every name and number kept, and a question still asked.
        let kept = need.iter().all(|w| low.contains(w.as_str()))
            && (!content.trim_end().ends_with('?') || l.contains('?'))
            // No question of its own: no unprompted small talk.
            && (content.contains('?') || !l.contains('?'));
        let words = l.split_whitespace().count();
        if !kept
            || words == 0
            || words > CHAT_WORDS + 6
            || crate::voice::leaks_instructions(&l)
            || crate::voice::leaks_example(&l, &real)
            || l.contains('[')
            || self.said.repeats(&l)
        {
            tracing::info!(line = l, content, "rephrase rejected; plain line");
            return Ok(content.to_owned());
        }
        Ok(l)
    }

    /// Collect `stream` until it ends or `deadline`, saying the first
    /// sentence as soon as it is complete when it is sayable (at least
    /// three words, passes the guards) and `first_ok`. Returns the raw text
    /// (`None` when late or failed) and the said sentence with how many
    /// raw bytes it covered. `Err` = cancelled (barge-in), even after the
    /// first sentence went out: the speaker stops it.
    async fn stream_say_first(
        &mut self,
        mut stream: crate::backend::EventStream,
        deadline: tokio::time::Instant,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: &CancellationToken,
        first_ok: &(dyn Fn(&str) -> bool + Send + Sync),
    ) -> Result<(Option<String>, Option<(String, usize)>), ()> {
        use futures_util::StreamExt as _;
        let mut text = String::new();
        let mut first: Option<(String, usize)> = None;
        let mut tried = false;
        // A model reply that is slow to start gets a short natural
        // acknowledgement first, the way a person says "Mm," while thinking.
        let ack_at = tokio::time::Instant::now() + ACK_AFTER;
        let mut acked = !self.people.as_ref().is_some_and(|p| p.think_ack);
        let raw = loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(()),
                () = tokio::time::sleep_until(deadline) => break None,
                () = tokio::time::sleep_until(ack_at), if !acked && first.is_none() => {
                    acked = true;
                    let n = ACK_TURN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let ack = ACKS[n % ACKS.len()];
                    tracing::info!(ack, "thinking aloud");
                    self.say(ack.to_owned());
                }
                o = obs.recv() => {
                    if let Some(o) = o
                        && barge_in(&o)
                    {
                        return Err(());
                    }
                }
                ev = stream.next() => match ev {
                    None => break Some(text.clone()),
                    Some(Err(_)) => break None,
                    Some(Ok(ChatEvent::Text(t))) => {
                        text.push_str(&t);
                        if !tried && let Some(end) = sentence_end(&text) {
                            tried = true;
                            let said = crate::voice::speakable(&text[..end], CHAT_WORDS)
                                .filter(|s| s.split_whitespace().count() >= 3)
                                .filter(|s| self.line_ok(s) && first_ok(s));
                            if let Some(s) = said {
                                tracing::info!(sentence = s, "said while thinking");
                                self.say(s.clone());
                                first = Some((s, end));
                            }
                        }
                    }
                    Some(Ok(ChatEvent::Call(_))) => {}
                }
            }
        };
        Ok((raw, first))
    }

    /// The line a streamed reply ends as: the sentence already said, plus
    /// at most one more within [`CHAT_WORDS`], said now if it passes the
    /// guards (and carries nothing of `need` that the first lacked... all
    /// of `need` was in the first, so nothing more is required).
    fn finish_streamed(
        &mut self,
        first: (String, usize),
        raw: Option<&str>,
        need: &[String],
    ) -> String {
        let _ = need;
        let (said, used) = first;
        let mut line = said.clone();
        let left = CHAT_WORDS.saturating_sub(said.split_whitespace().count());
        if let Some(rest) = raw.and_then(|r| r.get(used..))
            && left >= 3
            && let Some(more) = crate::voice::speakable(rest, CHAT_WORDS)
        {
            let more = crate::voice::first_sentence(&more);
            if !more.is_empty() && more.split_whitespace().count() <= left && self.line_ok(&more) {
                self.say(more.clone());
                line = format!("{line} {more}");
            }
        }
        self.streamed = true;
        line
    }

    /// The per-sentence guards: no leaked instructions or example names,
    /// no prompt brackets, no repeat of a recent line.
    fn line_ok(&self, l: &str) -> bool {
        let real = self.real_context();
        !crate::voice::leaks_instructions(l)
            && !crate::voice::leaks_example(l, &real)
            && !l.contains('[')
            && !self.said.repeats(l)
    }

    async fn chat_line(
        &mut self,
        text: &str,
        me: Option<&EntityId>,
        name: Option<&str>,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: &CancellationToken,
    ) -> Result<Option<String>, ()> {
        // Chat gets the name and the conversation only: what the bot knows
        // about them (likes, family, past visits) is said when they ask,
        // never volunteered.
        let _ = me;
        let ctx = match name {
            Some(n) => format!("[talking with] {n}"),
            None => "[talking with] someone new".to_owned(),
        };
        let req = ChatRequest {
            messages: {
                // The conversation so far as real turns, so the reply
                // follows on from it, then the person card and the line.
                let mut m = vec![Message::system(format!(
                    "{PERSONA}
{}",
                    now_line()
                ))];
                for e in self.visit.recent() {
                    m.push(Message::user(e.heard.clone()));
                    m.push(Message::assistant(e.said.clone()));
                }
                m.push(Message::user(format!("{ctx}\n[said] {text}")));
                m
            },
            tools: Vec::new(),
            max_tokens: 60,
            temperature: 0.7,
            json_object: false,
        };
        let stream = self.backend.chat(req);
        // Bounded: a slow or stuck model gives way to a short line of our own.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
        // Think while speaking: the first sentence is said the moment it
        // is complete, while the model writes the rest.
        let (raw, first) = self
            .stream_say_first(stream, deadline, obs, cancel, &|_| true)
            .await?;
        if let Some(first) = first {
            return Ok(Some(self.finish_streamed(first, raw.as_deref(), &[])));
        }
        let raw = raw.unwrap_or_default();
        let mut l = crate::voice::speakable(&raw, CHAT_WORDS).unwrap_or_default();
        // The guards: no leaked instructions, no invented example names,
        // no repeat of the last line.
        let real = self.real_context();
        if crate::voice::leaks_instructions(&l)
            || crate::voice::leaks_example(&l, &real)
            || (!l.is_empty() && self.said.repeats(&l))
        {
            tracing::info!(line = l, "chat line dropped by a guard");
            l = String::new();
        }
        let words: Vec<&str> = l.split_whitespace().collect();
        if words.len() > CHAT_WORDS {
            l = crate::voice::first_sentence(&l);
        }
        if l.trim().is_empty() {
            l = match name {
                Some(n) => format!("Tell me more, {n}!"),
                None => "Ooh, tell me more!".to_owned(),
            };
        }
        Ok(Some(l))
    }
}

/// An understander confidence below which an open question goes to the
/// slow path.
const DEEP_BELOW: f32 = 0.7;
/// The slow path's whole budget, holding line to answer.
const DEEP_CAP: Duration = Duration::from_secs(6);
/// Word ceiling of a slow-path answer.
const DEEP_WORDS: usize = 50;
/// Said at once when a question needs thinking, taken in turn.
pub const HOLDS: &[&str] = &[
    "Hmm, let me think about that.",
    "Good question. Let me think.",
    "Ooh, give me a second to think.",
    "Let me think about that for a moment.",
    "Hmm, let me put that together.",
];
static HOLD_TURN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// When the slow path runs out of time or the model fails.
pub const DEEP_GIVE_UP: &str =
    "Sorry, I couldn't think that one through just now. Could you ask me again?";
/// The slow path's instructions.
const DEEP_RULES: &str = "Now a question that needs real thought. First think briefly inside <think></think>, never spoken, in at most three short lines: what exactly they ask; which facts above (their people, what they told you, this visit, the times and details) decide it; the answer a kind, sensible friend would give. Then, after </think>, the spoken answer: one to three short sentences, at most 45 words, warm and specific, using what they told you. Never invent people, facts or events you were not told. If the answer is not in what you know, say so simply and suggest who could help (their family, a teacher, the front desk). No emoji, no lists.";

impl Session {
    /// The slow path: a holding line at once, then the stronger model with
    /// the whole layered context and a silent reasoning step, within
    /// [`DEEP_CAP`]. `Err` = cancelled (barge-in).
    async fn deep_line(
        &mut self,
        text: &str,
        me: Option<&EntityId>,
        name: Option<&str>,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: &CancellationToken,
    ) -> Result<String, ()> {
        use futures_util::StreamExt as _;
        let t0 = tokio::time::Instant::now();
        let n = HOLD_TURN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let hold = HOLDS[n % HOLDS.len()];
        tracing::info!(hold, "slow path: holding line");
        self.say(hold.to_owned());
        self.said.push(hold);
        // Layers: who they are (profile, relations, facts), their people's
        // own facts, the visit notebook, then the last turns as real turns.
        let mut ctx = converse::layered(self.facts.as_ref(), me, name, &self.visit);
        if let Some(id) = me {
            for e in self.facts.edges(id).iter().take(6) {
                let known: Vec<String> = self
                    .facts
                    .recall(&e.other)
                    .into_iter()
                    .filter(|f| converse::shown(f))
                    .take(4)
                    .collect();
                if !known.is_empty() {
                    ctx.push_str(&format!("\n[about {}] {}", e.other_name, known.join("; ")));
                }
            }
        }
        let mut m = vec![Message::system(format!(
            "{PERSONA}\n{}\n{ctx}\n{DEEP_RULES}",
            now_line()
        ))];
        for e in self.visit.recent() {
            m.push(Message::user(e.heard.clone()));
            m.push(Message::assistant(e.said.clone()));
        }
        let who = name.map(|n| format!(" ({n})")).unwrap_or_default();
        m.push(Message::user(format!("[they ask]{who} {text}")));
        let req = ChatRequest {
            messages: m,
            tools: Vec::new(),
            max_tokens: 220,
            temperature: 0.4,
            json_object: false,
        };
        let backend = self
            .people
            .as_ref()
            .and_then(|p| p.deep.clone())
            .unwrap_or_else(|| Arc::clone(&self.backend));
        let mut stream = backend.chat(req);
        let deadline = t0 + DEEP_CAP;
        let mut raw = String::new();
        let done = loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(()),
                () = tokio::time::sleep_until(deadline) => break false,
                o = obs.recv() => {
                    if let Some(o) = o
                        && barge_in(&o)
                    {
                        return Err(());
                    }
                }
                ev = stream.next() => match ev {
                    None => break true,
                    Some(Err(e)) => {
                        tracing::info!(error = %e, "slow path: model failed");
                        break false;
                    }
                    Some(Ok(ChatEvent::Text(t))) => raw.push_str(&t),
                    Some(Ok(ChatEvent::Call(_))) => {}
                }
            }
        };
        let ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
        let (mut answer, reasoning) = crate::voice::strip_reasoning(&raw);
        // Out of time mid-answer: the complete sentences so far are said.
        let partial = !done && raw.contains("</think>");
        if partial {
            let cut = answer.rfind(['.', '!', '?']).map_or(0, |i| i + 1);
            answer.truncate(cut);
        }
        let line = if done || partial {
            // The guards, with everything the prompt carried counted as
            // real: names from the graph are not invented ones.
            let real = format!("{} {}", self.real_context(), ctx.to_lowercase());
            deep_answer(&answer).filter(|l| {
                !crate::voice::leaks_example(l, &real) && !l.contains('[') && !self.said.repeats(l)
            })
        } else {
            None
        };
        tracing::info!(
            ms,
            done,
            reasoning = reasoning.as_str(),
            answer = line.as_deref().unwrap_or(""),
            "slow path answered"
        );
        Ok(line.unwrap_or_else(|| DEEP_GIVE_UP.to_owned()))
    }
}

/// The spoken part of a slow-path reply (the reasoning dropped, at most
/// two sentences and [`DEEP_WORDS`] words), or `None` when nothing sayable
/// is left.
fn deep_answer(answer: &str) -> Option<String> {
    crate::voice::speakable(answer, DEEP_WORDS)
        .map(|l| l.trim().to_owned())
        .filter(|l| !l.is_empty() && !crate::voice::leaks_instructions(l))
}

/// Lines whose exact words matter and are never reworded: privacy, the
/// hand-off, forgetting someone.
fn fixed_line(intent: &str, line: &str) -> bool {
    intent == "forget_me"
        || intent == "out_of_zone"
        || line == ASK_NAME
        || line.starts_with("I don't know")
        || line.starts_with("I only talk")
        || line == crate::handoff::DEFAULT_LINE
}

/// How long a model reply may take to start before an acknowledgement.
const ACK_AFTER: Duration = Duration::from_millis(600);
/// Short thinking sounds, taken in turn so none comes twice in a row.
const ACKS: &[&str] = &[
    "Mm,",
    "Oh!",
    "Hmm, let me see.",
    "Ah,",
    "Ooh,",
    "Right,",
    "Mm-hm,",
];
static ACK_TURN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Where the first complete sentence of a streamed `text` ends (after its
/// `.`, `!` or `?`, once the next character has arrived and is a space),
/// skipping titles like "Mr." and "Dr.".
fn sentence_end(text: &str) -> Option<usize> {
    let b = text.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if !matches!(c, b'.' | b'!' | b'?') {
            continue;
        }
        if !b.get(i + 1).is_some_and(u8::is_ascii_whitespace) {
            continue;
        }
        let word = text[..i].rsplit(char::is_whitespace).next().unwrap_or("");
        if c == b'.' && matches!(word, "Mr" | "Mrs" | "Ms" | "Dr" | "St" | "Prof") {
            continue;
        }
        return Some(i + 1);
    }
    None
}

/// The words a rephrase must keep from `content`: names and numbers,
/// lower-cased.
fn required_words(content: &str) -> Vec<String> {
    // What "You told me ..." reports must survive too: "hospital".
    let told: Vec<String> = content
        .split("told me")
        .skip(1)
        .flat_map(|t| t.split(|c: char| !c.is_alphanumeric()))
        .filter(|w| {
            w.len() >= 4
                && !matches!(
                    *w,
                    "your"
                        | "that"
                        | "with"
                        | "have"
                        | "work"
                        | "works"
                        | "like"
                        | "likes"
                        | "favourite"
                )
        })
        .map(str::to_lowercase)
        .collect();
    let mut need = named_words(content);
    need.extend(told);
    // The relation said must survive: "Sunita is your wife".
    for w in content.split(|c: char| !c.is_alphanumeric()) {
        let lw = w.to_lowercase();
        if crate::people::Relation::from_word(&lw).is_some() && lw.len() > 3 && !need.contains(&lw)
        {
            need.push(lw);
        }
    }
    need
}

fn named_words(content: &str) -> Vec<String> {
    content
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            !w.is_empty()
                && (w.chars().next().is_some_and(char::is_uppercase) && w.len() > 1 && *w != "I"
                    || w.chars().all(|c| c.is_ascii_digit()))
        })
        .filter(|w| {
            !matches!(
                *w,
                "Hi" | "Hello"
                    | "Bye"
                    | "You"
                    | "I'm"
                    | "Okay"
                    | "Oh"
                    | "Good"
                    | "Nice"
                    | "That"
                    | "Yes"
                    | "No"
                    | "Got"
                    | "Glad"
                    | "Thanks"
                    | "Sorry"
                    | "The"
                    | "It"
                    | "See"
                    | "Tell"
                    | "What"
                    | "How"
                    | "Who"
                    | "Your"
                    | "They"
                    | "He"
                    | "She"
            )
        })
        .map(str::to_lowercase)
        .collect()
}

/// "likes cricket" -> "like cricket"; "has a dog" -> "have a dog".
/// The content words of a question about the speaker, when it asks for a
/// detail rather than `who am I`: `what is my favourite colour?` gives
/// `["favourite", "favorite", "colour", "color"]`. `None` for name/summary
/// questions.
fn asked_detail(text: &str) -> Option<Vec<String>> {
    const STOP: &[&str] = &[
        "what", "whats", "what's", "is", "are", "my", "the", "a", "an", "do", "you", "know",
        "remember", "about", "me", "i", "of", "tell", "who", "am", "name", "hey", "please", "can",
        "your", "did", "say", "said", "to", "what", "which",
    ];
    let mut words: Vec<String> = text
        .to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.len() > 2 && !STOP.contains(w))
        .map(str::to_owned)
        .collect();
    if words.is_empty() {
        return None;
    }
    // British and American spellings both match the stored fact.
    for (a, b) in [("favourite", "favorite"), ("colour", "color")] {
        if words.iter().any(|w| w == a) {
            words.push(b.to_owned());
        } else if words.iter().any(|w| w == b) {
            words.push(a.to_owned());
        }
    }
    Some(words)
}

/// "Good morning" / "Good afternoon" / "Good evening" by the local hour.
fn time_greeting() -> &'static str {
    match local_hour() {
        5..=11 => "Good morning",
        12..=16 => "Good afternoon",
        _ => "Good evening",
    }
}

/// The local hour, 0..24.
fn local_hour() -> i64 {
    super::local_clock()
        .0
        .split(':')
        .next()
        .and_then(|h| h.parse().ok())
        .unwrap_or(12)
}

/// A remark on how long it's been, when the gap is worth one.
fn since_phrase(gap_secs: f64) -> Option<String> {
    let days = gap_secs / 86_400.0;
    Some(if days >= 30.0 {
        "It's been ages! Lovely to see you.".to_owned()
    } else if days >= 7.0 {
        format!("It's been {} weeks!", (days / 7.0).round() as i64).replace("1 weeks", "a week")
    } else if days >= 2.0 {
        format!("Haven't seen you for {} days!", days.round() as i64)
    } else if days >= 0.6 {
        "Nice to see you again!".to_owned()
    } else {
        return None;
    })
}

/// The time line every model prompt carries: "[now] Monday 12 October, 14:20 (afternoon)".
fn now_line() -> String {
    let (time, date) = super::local_clock();
    let part = match local_hour() {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=20 => "evening",
        _ => "night",
    };
    format!("[now] {date}, {time} ({part})")
}

/// The time, the date, a sum or a spelling, worked out here.
fn everyday_answer(text: &str) -> String {
    let t = text.to_ascii_lowercase();
    if let Some(v) = super::sum(&t) {
        return format!("That's {v}.");
    }
    let words: Vec<&str> = t
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if let Some(w) = crate::templates::spelling_word(&words) {
        let letters: Vec<String> = w
            .chars()
            .map(|c| c.to_ascii_uppercase().to_string())
            .collect();
        return format!("{}.", letters.join("-"));
    }
    if t.contains("tomorrow") {
        return format!("Tomorrow is {}.", super::local_clock_tomorrow());
    }
    let (time, date) = super::local_clock();
    if t.contains("time") {
        format!("It's {time}.")
    } else {
        format!("It's {date}.")
    }
}

/// The name a correction gives when no rule found it: the last
/// capitalised word that is not the name held now ("sorry, I'm Neela, the
/// line was noisy", "not Meena, Mina").
fn said_name(text: &str, held: Option<&str>) -> Option<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|w| w.len() >= 2 && w.chars().next().is_some_and(char::is_uppercase))
        .filter(|w| {
            !matches!(
                *w,
                "I" | "No" | "Not" | "Sorry" | "It" | "My" | "That" | "Its" | "Glydi"
            )
        })
        .filter(|w| held.is_none_or(|h| !h.eq_ignore_ascii_case(w)))
        .find(|w| crate::voice::plausible_name(w))
        .map(str::to_owned)
}

/// "What's my name?", "who am I?", "do you know me?".
fn asks_name_only(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    ["my name", "who am i", "know me", "remember me"]
        .iter()
        .any(|p| t.contains(p))
}

fn second_person(fact: &str) -> String {
    let f = fact.trim().trim_end_matches('.');
    if f.starts_with("favourite ") {
        return format!("your {f}");
    }
    let mut w: Vec<String> = f.split_whitespace().map(str::to_owned).collect();
    if let Some(first) = w.first_mut() {
        *first = match first.as_str() {
            "has" => "have".to_owned(),
            "is" => "are".to_owned(),
            x if x.ends_with('s') && x.len() > 3 => x[..x.len() - 1].to_owned(),
            x => x.to_owned(),
        };
    }
    w.join(" ")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::time::Instant;

    use common::{CommandQueue, FakeClock};
    use mind::{ViewEntity, WorldView};
    use parking_lot::Mutex;

    use super::*;
    use crate::deliberator::{Config, Snapshot};
    use crate::mock::{MockLlm, Script};
    use crate::tools::{FactSource, InMemoryFacts};
    use crate::understand::ScriptedUnderstander;

    fn face(id: &str, name: Option<&str>) -> ViewEntity {
        ViewEntity {
            id: EntityId::new(id),
            name: name.map(Into::into),
            confidence: 0.9,
            is_speaking: false,
            first_seen: Instant::now(),
            returned: None,
        }
    }

    struct Rig {
        s: Session,
        q: Arc<CommandQueue>,
        facts: Arc<InMemoryFacts>,
        room: Arc<Mutex<Vec<ViewEntity>>>,
        model: Arc<ScriptedUnderstander>,
        rx: mpsc::Receiver<Observation>,
        _tx: mpsc::Sender<Observation>,
    }

    fn rig(chat: Vec<Script>) -> Rig {
        rig_with(chat, Arc::new(|_| {}))
    }

    fn rig_with(chat: Vec<Script>, expect_hook: ExpectName) -> Rig {
        let room: Arc<Mutex<Vec<ViewEntity>>> = Arc::default();
        let r2 = Arc::clone(&room);
        let snapshot: Snapshot = Box::new(move || {
            Arc::new(WorldView {
                at: Instant::now(),
                people: r2.lock().clone(),
                bot_speaking: false,
                working: mind::WorkingSnapshot::default(),
            })
        });
        let model = Arc::new(ScriptedUnderstander::default());
        let facts = Arc::new(InMemoryFacts::new());
        let q = Arc::new(CommandQueue::new());
        let config = Config {
            attention: false,
            people: Some(People {
                understander: Some(model.clone()),
                handoff: Arc::new(crate::handoff::DefaultHandoff),
                think_ack: true,
                deep: None,
                expecting_name: Some(expect_hook),
            }),
            ..Config::default()
        };
        let s = Session::new(
            MockLlm::new(chat),
            config,
            snapshot,
            facts.clone(),
            q.clone(),
            Arc::new(FakeClock::new()),
        );
        let (tx, rx) = mpsc::channel(1);
        Rig {
            s,
            q,
            facts,
            room,
            model,
            rx,
            _tx: tx,
        }
    }

    impl Rig {
        async fn say(&mut self, text: &str, speaker: Option<&EntityId>) -> String {
            while self.q.try_pop().is_some() {}
            self.s
                .handle_utterance(text, speaker, &mut self.rx, CancellationToken::new())
                .await
                .unwrap();
            std::iter::from_fn(|| self.q.try_pop())
                .filter(|c| c.kind == "say")
                .map(|c| c.payload.as_text().unwrap_or("").to_owned())
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    fn json(intent: &str, people: &str, facts: &str) -> String {
        format!(
            r#"{{"intent":"{intent}","people":{people},"facts":{facts},"refers_to":null,"heard_ok":true,"confidence":0.9}}"#
        )
    }

    #[tokio::test]
    async fn father_then_is_my_son_here() {
        let mut r = rig(vec![]);
        let line = r.say("I'm Suresh, Rohan's father", None).await;
        assert_eq!(
            line,
            "Nice to meet you, Suresh! I'll remember you're Rohan's father."
        );
        let suresh = r.facts.resolve_name("Suresh").unwrap();
        let rohan = r.facts.resolve_name("Rohan").unwrap();
        r.facts.saw(&rohan, unix_now() - 60.0);
        let line = r.say("is my son here?", Some(&suresh)).await;
        assert_eq!(line, "Yes, Rohan came in today.");
        // The pronoun now points at Rohan.
        let line = r.say("how is he?", Some(&suresh)).await;
        assert!(line.contains("Rohan"), "{line}");
        // A stranger does not get Rohan's day.
        let meena = r.facts.add_person("Meena");
        r.s.new_visit();
        r.s.current_person = None;
        let line = r.say("is Rohan here?", Some(&meena)).await;
        assert!(line.starts_with("I only talk"), "{line}");
    }

    #[tokio::test]
    async fn sister_beside_is_named_and_greeted_next_visit() {
        let mut r = rig(vec![]);
        let sony = r.facts.add_person("Sony");
        *r.room.lock() = vec![face(sony.as_str(), Some("Sony")), face("track:7", None)];
        let line = r.say("this is my sister Priya", Some(&sony)).await;
        assert_eq!(line, "Hi Priya! Nice to meet you.");
        let priya = r.facts.resolve_name("Priya").unwrap();
        let edges = r.facts.edges(&priya);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation, Relation::SiblingOf);
        assert_eq!(edges[0].other, sony);
        // Next visit: Priya alone, greeted by name -- what the bot knows
        // about her is said only when she asks.
        *r.room.lock() = vec![face(priya.as_str(), Some("Priya"))];
        r.s.new_visit();
        let g = r.s.people_greeting(&priya, "Priya");
        assert!(g.contains("Priya") && !g.contains("Sony"), "{g}");
    }

    #[tokio::test]
    async fn pronoun_from_the_model_resolves_from_the_notebook() {
        let mut r = rig(vec![]);
        let asha = r.facts.add_person("Asha");
        r.facts
            .relate_people(&asha, &Relation::ParentOf, "Diya", "x", 0.9);
        r.model.push(&json(
            "tell_fact",
            r#"[{"name":"Diya","relation_to_speaker":"daughter"}]"#,
            "[]",
        ));
        r.say("Diya started dancing classes", Some(&asha)).await;
        let line = r.say("how is she?", Some(&asha)).await;
        assert!(line.contains("Diya"), "{line}");
    }

    #[tokio::test]
    async fn forget_me_forgets() {
        let mut r = rig(vec![]);
        r.say("my name is Kiran", None).await;
        let kiran = r.facts.resolve_name("Kiran").unwrap();
        let line = r.say("what's my name?", Some(&kiran)).await;
        assert!(line.starts_with("You're Kiran"), "{line}");
        let line = r.say("please forget me", Some(&kiran)).await;
        assert!(line.contains("forgotten"));
        assert!(r.facts.resolve_name("Kiran").is_none());
        let line = r.say("what's my name?", None).await;
        assert!(line.contains("don't know your name"), "{line}");
    }

    #[tokio::test]
    async fn out_of_zone_hands_off_and_emergency_is_fixed() {
        let mut r = rig(vec![]);
        let line = r.say("what is the capital of France?", None).await;
        assert_eq!(line, crate::handoff::DEFAULT_LINE);
        let line = r.say("my friend is bleeding", None).await;
        assert!(line.contains("teacher"), "{line}");
    }

    /// Think while speaking: the first sentence goes to the speaker while
    /// the model is still writing the second.
    #[tokio::test]
    async fn first_sentence_is_said_before_the_stream_ends() {
        let mut script = Script::text(&[
            "A red bike with stripes! ",
            "Do",
            " you",
            " ride",
            " it",
            " to",
            " school?",
        ]);
        script.delay = Duration::from_millis(150);
        let mut r = rig(vec![script]);
        let a = r.facts.add_person("Om");
        r.model.push(&json("chit_chat", "[]", "[]"));
        while r.q.try_pop().is_some() {}
        let q = Arc::clone(&r.q);
        let started = tokio::time::Instant::now();
        let watch = async move {
            loop {
                if let Some(c) = q.try_pop()
                    && c.kind == "say"
                {
                    return (
                        started.elapsed(),
                        c.payload.as_text().unwrap_or("").to_owned(),
                    );
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        };
        let (done_at, (first_at, first)) = tokio::join!(
            async {
                r.s.handle_utterance(
                    "I got a new bike, red with stripes",
                    Some(&a),
                    &mut r.rx,
                    CancellationToken::new(),
                )
                .await
                .unwrap();
                started.elapsed()
            },
            watch
        );
        assert_eq!(first, "A red bike with stripes!");
        // Six more fragments at 150 ms each came after the first sentence
        // was out: it was said while the model was still writing.
        assert!(
            done_at.saturating_sub(first_at) >= Duration::from_millis(700),
            "first at {first_at:?}, done at {done_at:?}"
        );
        let rest: Vec<String> = std::iter::from_fn(|| r.q.try_pop())
            .filter(|c| c.kind == "say")
            .map(|c| c.payload.as_text().unwrap_or("").to_owned())
            .collect();
        assert_eq!(rest, vec!["Do you ride it to school?".to_owned()]);
    }

    /// A slow model start gets a short acknowledgement first.
    #[tokio::test]
    async fn slow_reply_is_acknowledged_first() {
        let mut script = Script::text(&["That sounds like a lovely trip. ", "Who", " goes?"]);
        script.delay = Duration::from_millis(800);
        let mut r = rig(vec![script]);
        let a = r.facts.add_person("Om");
        r.model.push(&json("chit_chat", "[]", "[]"));
        let all = r.say("we are going to the hills next week", Some(&a)).await;
        let first = all.split(' ').next().unwrap_or("");
        assert!(ACKS.iter().any(|k| k.starts_with(first)), "{all}");
        assert!(all.contains("That sounds like a lovely trip."), "{all}");
    }

    /// The model dies mid-turn (Ollama restarted), or writes junk: a short
    /// line of our own, no hang.
    #[tokio::test]
    async fn model_failure_mid_turn_gives_a_short_line() {
        let broken = Script {
            events: vec![
                Ok(ChatEvent::Text("Oh I".to_owned())),
                Err(LlmError::Server {
                    status: 500,
                    body: "restarting".to_owned(),
                }),
            ],
            delay: Duration::ZERO,
        };
        let junk = Script::text(&["**[note]** \u{1F600}"]);
        let foreign = Script::text(&["\u{4F60}\u{597D}\u{FF01}\u{5F88}\u{9AD8}\u{5174}"]);
        let mut r = rig(vec![broken, junk, foreign]);
        let a = r.facts.add_person("Asha");
        for _ in 0..3 {
            r.model.push(&json("chit_chat", "[]", "[]"));
            let line = r.say("my day was a bit odd", Some(&a)).await;
            assert_eq!(line, "Tell me more, Asha!");
        }
    }

    #[tokio::test]
    async fn facts_add_update_and_chat_uses_the_model() {
        let mut r = rig(vec![Script::text(&["Long days are tiring. Rest well!"])]);
        let a = r.facts.add_person("Asha");
        assert!(
            !r.say("my favourite colour is blue", Some(&a))
                .await
                .is_empty()
        );
        assert_eq!(
            r.say("actually my favourite colour is red", Some(&a)).await,
            "Got it, I've updated that."
        );
        assert_eq!(
            r.facts.recall(&a),
            vec!["favourite colour is red".to_owned()]
        );
        r.model.push(&json("feeling", "[]", "[]"));
        let line = r.say("I had such a long day", Some(&a)).await;
        assert_eq!(line, "Long days are tiring. Rest well!");
        // Bad JSON twice: one clarifying question.
        r.model.push("{");
        r.model.push("{");
        let line = r.say("blorp the flimble", Some(&a)).await;
        assert!(CLARIFY.contains(&line.as_str()), "{line}");
    }

    /// The hearing side is told when a bare reply may be a name.
    #[tokio::test]
    async fn name_expected_while_the_bot_waits_for_one() {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f2 = Arc::clone(&flag);
        let mut r = rig_with(
            vec![],
            Arc::new(move |on| f2.store(on, std::sync::atomic::Ordering::Relaxed)),
        );
        let on = || flag.load(std::sync::atomic::Ordering::Relaxed);
        r.say("hello", None).await;
        assert!(on(), "asked for a name");
        r.say("Meera", None).await;
        assert!(!on(), "name given");
        let meera = r.facts.resolve_name("Meera").unwrap();
        r.say("this is my sister", Some(&meera)).await;
        assert!(on(), "asked for the sister's name");
        r.say("Priya", Some(&meera)).await;
        assert!(!on());
    }

    #[tokio::test]
    async fn router_sends_thinking_questions_to_the_slow_path() {
        let mut r = rig(vec![Script::text(&[
            "<think>Priya loves painting; a gift for painting.</think>",
            "Priya loves painting, so a set of brushes would be lovely.",
        ])]);
        let kavya = r.facts.add_person("Kavya");
        r.facts
            .relate_people(&kavya, &Relation::SiblingOf, "Priya", "x", 0.9);
        let line = r
            .say(
                "what should I get my sister for her birthday?",
                Some(&kavya),
            )
            .await;
        assert!(HOLDS.iter().any(|h| line.starts_with(h)), "{line}");
        assert!(
            line.ends_with("a set of brushes would be lovely."),
            "{line}"
        );
        assert!(!line.contains("think>"), "{line}");
        // A plain question stays fast.
        let line = r.say("what's my name?", Some(&kavya)).await;
        assert!(line.starts_with("You're Kavya"), "{line}");
    }

    #[tokio::test]
    async fn slow_path_gives_a_graceful_line_when_the_model_is_late() {
        let mut late = Script::text(&["<think>hmm</think> Too late."]);
        late.delay = Duration::from_secs(7);
        let mut r = rig(vec![late]);
        let a = r.facts.add_person("Om");
        let line = r.say("who should I ask about my cough?", Some(&a)).await;
        assert!(line.ends_with(DEEP_GIVE_UP), "{line}");
    }

    #[tokio::test]
    async fn stranger_greeting_asks_name_and_takes_a_bare_name() {
        let mut r = rig(vec![]);
        assert_eq!(r.say("hello", None).await, ASK_NAME);
        assert_eq!(r.say("Meera", None).await, "Nice to meet you, Meera!");
    }
}
