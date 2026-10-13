//! DECIDE + ANSWER: code, not the model, answers from the people graph,
//! the facts and the visit notebook. Short fixed lines, rotated. The
//! model writes only chit-chat (see the session glue,
//! `deliberator::people_turn`), with the layered prompt built here.

#![allow(clippy::format_push_string)]

use common::EntityId;

use crate::people::{Edge, FactOp, Relation, may_discuss};
use crate::tools::FactSource;
use crate::understand::Mention;
use crate::visit::{LAST_VISIT, Visit};

/// Seen within this many seconds counts as "here today".
pub const TODAY_SECS: f64 = 12.0 * 3600.0;

/// The fixed persona: layer (i) of every model prompt.
pub const PERSONA: &str = "You are Glydi, a warm, cheerful robot that greets people at the door. You talk with everyone who comes by, children and adults, visitors and staff, like a kind friend who remembers them. How you talk:
- React to what they actually said, specifically, using their own words.
- Do NOT bring up what you know about them (their likes, family, past visits) unless they ask you about it. Just use their name now and then.
- Show feeling: be glad with them, gentle when they are sad or nervous, playful when they joke.
- Keep it going: end with one short, natural question about THEM or their people, not about facts.
- One or two short sentences, at most 22 words, spoken out loud, no emoji, no lists.
- You may answer simple everyday things anyone knows (animals, colours, the body, simple why-questions) in one short sentence. Bigger knowledge questions (history, geography, famous people, news, the business of this place) you don't know; say so kindly and suggest asking someone at the front desk.
- Match the person: playful with children, warm and polite with adults.
- Never invent names, facts or events. Never say just \"That's nice\".
Examples:
They: I'm nervous about my maths test. You: Oh, tests can feel big! You've practised, right? Which part feels hardest?
They: I scored a goal today! You: A goal! Brilliant! Was it a big kick or a sneaky one?
They: I'm bored. You: Uh oh, a bored mind! What's your favourite thing to do at break?
They: Nothing much. You: A quiet day then! What was the best bit so far?
They: My grandma is sick. You: I'm sorry to hear that. I hope she feels better soon. Are you okay?";

/// Pick one of `options` by `salt`, never the same as `last`.
pub fn pick(options: &[&str], salt: usize, last: Option<&str>) -> String {
    let n = options.len().max(1);
    for step in 0..n {
        let o = options[(salt + step) % n];
        if Some(o) != last {
            return o.to_owned();
        }
    }
    options[salt % n].to_owned()
}

/// "Rohan's {word}" for an edge seen from the person being described.
pub fn relation_noun(r: &Relation) -> &str {
    match r {
        Relation::ParentOf => "parent",
        Relation::ChildOf => "child",
        Relation::SiblingOf => "sibling",
        Relation::FriendOf => "friend",
        Relation::TeacherOf => "teacher",
        Relation::StudentOf => "student",
        Relation::SpouseOf => "family",
        Relation::Other(w) => w,
    }
}

/// Layer (ii): the long-term profile of the speaker.
pub fn profile(facts: &dyn FactSource, speaker: Option<&EntityId>, name: Option<&str>) -> String {
    let Some(id) = speaker else {
        return "[person] Someone new; you do not know their name.".to_owned();
    };
    let mut s = format!("[person] {}", name.unwrap_or("Someone known"));
    let edges = facts.edges(id);
    if !edges.is_empty() {
        let rel: Vec<String> = edges
            .iter()
            .map(|e| {
                format!(
                    "{} is their {}",
                    e.other_name,
                    relation_noun(&e.relation.inverse())
                )
            })
            .collect();
        s.push_str(". ");
        s.push_str(&rel.join("; "));
    }
    let all = facts.recall(id);
    let plain: Vec<&String> = all.iter().filter(|f| !f.starts_with(LAST_VISIT)).collect();
    if !plain.is_empty() {
        s.push_str(". Known: ");
        s.push_str(
            &plain
                .iter()
                .rev()
                .take(5)
                .map(|f| f.as_str())
                .collect::<Vec<_>>()
                .join("; "),
        );
    }
    if let Some(l) = all.iter().rev().find(|f| f.starts_with(LAST_VISIT)) {
        s.push_str(". ");
        s.push_str(l);
    }
    s.push('.');
    s
}

/// The whole layered context (ii)+(iii), the same shape every time.
pub fn layered(
    facts: &dyn FactSource,
    speaker: Option<&EntityId>,
    name: Option<&str>,
    visit: &Visit,
) -> String {
    format!("{}\n{}", profile(facts, speaker, name), visit.notebook())
}

/// Who a mention points at: a name, "my son", or a pronoun from the
/// notebook.
pub fn resolve(
    m: &Mention,
    refers_to: Option<&str>,
    speaker: Option<&EntityId>,
    facts: &dyn FactSource,
    visit: &Visit,
) -> Option<(EntityId, String)> {
    if !m.name.is_empty() {
        let id = facts.resolve_name(&m.name)?;
        let name = facts.name_of(&id).unwrap_or_else(|| m.name.clone());
        return Some((id, name));
    }
    if let (Some(word), Some(sp)) = (m.relation_to_speaker.as_deref(), speaker) {
        // "my son": the son is ChildOf me, so my side reads ParentOf.
        if let Some(r) = Relation::from_word(word) {
            let want = r.inverse();
            let edges = facts.edges(sp);
            if let Some(e) = edges.iter().find(|e| e.relation == want) {
                return Some((e.other.clone(), e.other_name.clone()));
            }
        }
    }
    let pronoun = m.reference.as_deref().or(refers_to)?;
    if matches!(
        pronoun.trim(),
        "she" | "he" | "her" | "him" | "they" | "them" | "that"
    ) {
        return visit.refs.last().cloned();
    }
    None
}

/// The answer to "how is Rohan?" / "is my son here?" / "who is she?".
#[allow(clippy::too_many_arguments)]
pub fn about_person(
    target: &(EntityId, String),
    speaker: Option<&EntityId>,
    facts: &dyn FactSource,
    now: f64,
    present: &[EntityId],
    who_question: bool,
) -> String {
    let (id, name) = target;
    let edges: Vec<Edge> = speaker.map(|s| facts.edges(s)).unwrap_or_default();
    if !may_discuss(speaker, id, &edges) {
        return "I only talk about people I know through you. The office can help.".to_owned();
    }
    let rel = edges
        .iter()
        .find(|e| &e.other == id)
        .map(|e| relation_noun(&e.relation.inverse()).to_owned());
    if who_question {
        return match rel {
            Some(r) => format!("That's {name}, your {r}."),
            None => format!("That's {name}."),
        };
    }
    let here = present.contains(id);
    let seen = facts.last_seen(id).filter(|t| now - t < TODAY_SECS);
    let mut line = if here {
        format!("{name} is right here!")
    } else if seen.is_some() {
        format!("Yes, {name} came in today.")
    } else {
        format!("I haven't seen {name} today.")
    };
    if let Some(f) = facts
        .recall(id)
        .iter()
        .rev()
        .find(|f| !f.starts_with(LAST_VISIT))
    {
        line.push_str(&format!(
            " Last I heard, {name} {}.",
            f.trim_end_matches('.')
        ));
    }
    line
}

/// What the person told about themselves, written with the fact
/// decision. Returns what happened, for the reply.
pub fn write_fact(facts: &dyn FactSource, id: &EntityId, fact: &str, retract: bool) -> FactOp {
    let held: Vec<String> = facts.recall(id);
    let op = crate::people::fact_op(&held, fact, retract);
    match &op {
        FactOp::Add => facts.remember(id, fact),
        FactOp::Update(old) => {
            facts.forget_fact(id, old);
            facts.remember(id, fact);
        }
        FactOp::Delete(old) => {
            facts.forget_fact(id, old);
        }
        FactOp::Noop => {}
    }
    tracing::info!(%id, fact, ?op, "fact write");
    op
}

/// Whether the utterance takes a fact back.
pub fn retracts(text: &str) -> bool {
    let t = text.to_lowercase();
    [
        "any more",
        "anymore",
        "no longer",
        "not true",
        "forget that",
        "that's wrong about me",
    ]
    .iter()
    .any(|p| t.contains(p))
}

/// The greeting for a known person arriving: a related person here today,
/// else a check-in on what was said last visit about someone, else their
/// relation, else plain. Once per visit (the caller gates).
pub fn greeting(
    id: &EntityId,
    name: &str,
    facts: &dyn FactSource,
    now: f64,
    present: &[EntityId],
) -> String {
    let edges = facts.edges(id);
    // Family seen today and not standing here: "Rohan's here today."
    if let Some(e) = edges.iter().find(|e| {
        matches!(
            e.relation,
            Relation::ParentOf | Relation::ChildOf | Relation::SiblingOf | Relation::SpouseOf
        ) && !present.contains(&e.other)
            && facts
                .last_seen(&e.other)
                .is_some_and(|t| now - t < TODAY_SECS)
    }) {
        return format!("Hi {name}! {}'s here today.", e.other_name);
    }
    // Last visit mentioned a relation and something to ask about.
    let all = facts.recall(id);
    if let Some(last) = all.iter().rev().find_map(|f| f.strip_prefix(LAST_VISIT))
        && let Some(e) = edges
            .iter()
            .find(|e| last.to_lowercase().contains(&e.other_name.to_lowercase()))
        && let Some(thing) = crate::understand::words(last)
            .into_iter()
            .find(|w| CHECK_WORDS.contains(&w.as_str()))
    {
        return format!("Hi {name}! How did {}'s {thing} go?", e.other_name);
    }
    if let Some(e) = edges.first() {
        let noun = relation_noun(&e.relation);
        return format!("Hi {name}! Nice to see {}'s {noun} again.", e.other_name);
    }
    format!("Hi {name}!")
}

/// What a last visit may be asked about.
pub const CHECK_WORDS: &[&str] = &[
    "match",
    "exam",
    "test",
    "trip",
    "game",
    "race",
    "show",
    "competition",
    "birthday",
    "interview",
    "doctor",
    "holiday",
    "party",
    "function",
    "recital",
    "tournament",
];

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::tools::InMemoryFacts;

    #[test]
    fn greeting_uses_the_graph() {
        let f = InMemoryFacts::new();
        let suresh = f.add_person("Suresh");
        let rohan = f
            .relate_people(&suresh, &Relation::ParentOf, "Rohan", "x", 0.9)
            .unwrap();
        assert_eq!(
            greeting(&suresh, "Suresh", &f, 100.0, &[]),
            "Hi Suresh! Nice to see Rohan's parent again."
        );
        f.saw(&rohan, 50.0);
        assert_eq!(
            greeting(&suresh, "Suresh", &f, 100.0, &[]),
            "Hi Suresh! Rohan's here today."
        );
        let f = InMemoryFacts::new();
        let s = f.add_person("Suresh");
        f.relate_people(&s, &Relation::ParentOf, "Rohan", "x", 0.9);
        f.remember(&s, "Last visit: Rohan has a cricket match on Sunday");
        assert_eq!(
            greeting(&s, "Suresh", &f, 100.0, &[]),
            "Hi Suresh! How did Rohan's match go?"
        );
    }

    #[test]
    fn about_person_respects_privacy() {
        let f = InMemoryFacts::new();
        let suresh = f.add_person("Suresh");
        let stranger = f.add_person("Meena");
        let rohan = f
            .relate_people(&suresh, &Relation::ParentOf, "Rohan", "x", 0.9)
            .unwrap();
        f.remember(&rohan, "likes cricket");
        f.saw(&rohan, 90.0);
        let t = (rohan.clone(), "Rohan".to_owned());
        assert_eq!(
            about_person(&t, Some(&suresh), &f, 100.0, &[], false),
            "Yes, Rohan came in today. Last I heard, Rohan likes cricket."
        );
        assert!(
            about_person(&t, Some(&stranger), &f, 100.0, &[], false).starts_with("I only talk")
        );
        assert!(about_person(&t, None, &f, 100.0, &[], false).starts_with("I only talk"));
        assert_eq!(
            about_person(&t, Some(&suresh), &f, 100.0, &[], true),
            "That's Rohan, your child."
        );
    }

    #[test]
    fn resolves_names_relations_and_pronouns() {
        let f = InMemoryFacts::new();
        let suresh = f.add_person("Suresh");
        let rohan = f
            .relate_people(&suresh, &Relation::ParentOf, "Rohan", "x", 0.9)
            .unwrap();
        let mut v = Visit::default();
        let my_son = Mention {
            relation_to_speaker: Some("son".into()),
            reference: Some("my son".into()),
            ..Mention::default()
        };
        assert_eq!(
            resolve(&my_son, None, Some(&suresh), &f, &v).map(|r| r.0),
            Some(rohan.clone())
        );
        let she = Mention {
            reference: Some("she".into()),
            ..Mention::default()
        };
        assert_eq!(resolve(&she, None, Some(&suresh), &f, &v), None);
        v.refer(rohan.clone(), "Rohan");
        assert_eq!(
            resolve(&she, None, Some(&suresh), &f, &v).map(|r| r.1),
            Some("Rohan".into())
        );
    }

    #[test]
    fn fact_writes() {
        let f = InMemoryFacts::new();
        let a = f.add_person("Asha");
        assert_eq!(
            write_fact(&f, &a, "favourite colour is blue", false),
            FactOp::Add
        );
        assert_eq!(
            write_fact(&f, &a, "favourite colour is blue", false),
            FactOp::Noop
        );
        assert_eq!(
            write_fact(&f, &a, "favourite colour is red", false),
            FactOp::Update("favourite colour is blue".into())
        );
        assert_eq!(f.recall(&a), vec!["favourite colour is red".to_owned()]);
        assert_eq!(
            write_fact(&f, &a, "favourite colour is red", true),
            FactOp::Delete("favourite colour is red".into())
        );
        assert!(f.recall(&a).is_empty());
        assert!(retracts("I don't like cricket any more"));
    }

    #[test]
    fn profile_is_layered() {
        let f = InMemoryFacts::new();
        let s = f.add_person("Suresh");
        f.relate_people(&s, &Relation::ParentOf, "Rohan", "x", 0.9);
        f.remember(&s, "likes tea");
        let p = profile(&f, Some(&s), Some("Suresh"));
        assert_eq!(
            p,
            "[person] Suresh. Rohan is their child. Known: likes tea."
        );
    }
}
