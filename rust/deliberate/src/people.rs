//! The people graph: who is related to whom, learned from talk.
//!
//! Edges are temporal (the Graphiti / Zep idea, kept small): each has a
//! `valid_from` and a `valid_to` (`None` = current), the utterance it came
//! from and a confidence. A conflicting new relation between the same two
//! people *closes* the old edge instead of overwriting it, and every query
//! reads current edges only. One direction is stored; the inverse
//! ("Rohan is Suresh's child" from "Suresh is Rohan's father") is derived.
//!
//! Also here, pure and shared by the in-memory source and the SQLite
//! store: the fact write decision ([`fact_op`], the Mem0 idea done with
//! rules) and the privacy rule ([`may_discuss`]).

#![allow(clippy::redundant_closure_for_method_calls, clippy::redundant_closure)]

use common::EntityId;

/// One kind of relation, from `a`'s side: `a` is `rel` of `b`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Relation {
    /// `a` is a parent of `b`.
    ParentOf,
    /// `a` is a child of `b`.
    ChildOf,
    /// Brother or sister.
    SiblingOf,
    /// Friend.
    FriendOf,
    /// `a` teaches `b`.
    TeacherOf,
    /// `a` is taught by `b`.
    StudentOf,
    /// Husband or wife.
    SpouseOf,
    /// Anything else, by its word ("cousin", "neighbour").
    Other(String),
}

impl Relation {
    /// The stored word.
    pub fn as_str(&self) -> &str {
        match self {
            Self::ParentOf => "parent_of",
            Self::ChildOf => "child_of",
            Self::SiblingOf => "sibling_of",
            Self::FriendOf => "friend_of",
            Self::TeacherOf => "teacher_of",
            Self::StudentOf => "student_of",
            Self::SpouseOf => "spouse_of",
            Self::Other(s) => s,
        }
    }

    /// Back from the stored word.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "parent_of" => Self::ParentOf,
            "child_of" => Self::ChildOf,
            "sibling_of" => Self::SiblingOf,
            "friend_of" => Self::FriendOf,
            "teacher_of" => Self::TeacherOf,
            "student_of" => Self::StudentOf,
            "spouse_of" => Self::SpouseOf,
            other => Self::Other(other.to_owned()),
        }
    }

    /// The same edge read from `b`'s side.
    #[must_use]
    pub fn inverse(&self) -> Self {
        match self {
            Self::ParentOf => Self::ChildOf,
            Self::ChildOf => Self::ParentOf,
            Self::TeacherOf => Self::StudentOf,
            Self::StudentOf => Self::TeacherOf,
            r => r.clone(),
        }
    }

    /// The relation of the *other* person to the speaker, from the word the
    /// speaker used for them: "my son" -> the son is `ChildOf` the speaker.
    /// `None` for a word that is not a relation.
    pub fn from_word(word: &str) -> Option<Self> {
        let w = word
            .trim()
            .trim_start_matches("my ")
            .trim_end_matches('s')
            .to_lowercase();
        Some(match w.as_str() {
            "son" | "daughter" | "child" | "kid" | "boy" | "girl" | "child_of" => Self::ChildOf,
            "father" | "mother" | "dad" | "mum" | "mom" | "parent" | "papa" | "mama"
            | "parent_of" => Self::ParentOf,
            "brother" | "sister" | "sibling" | "sibling_of" => Self::SiblingOf,
            "friend" | "best friend" | "friend_of" => Self::FriendOf,
            "teacher" | "teacher_of" => Self::TeacherOf,
            "student" | "pupil" | "student_of" => Self::StudentOf,
            "wife" | "husband" | "spouse" | "spouse_of" => Self::SpouseOf,
            "cousin" | "uncle" | "aunt" | "aunty" | "grandfather" | "grandmother" | "grandma"
            | "grandpa" | "neighbour" | "neighbor" | "nephew" | "niece" => Self::Other(w),
            _ => return None,
        })
    }

    /// How the relation reads in a sentence about `b` from `a`'s side,
    /// e.g. `ChildOf` -> "child". Used as "Rohan's {word}".
    pub fn word(&self) -> &str {
        match self {
            Self::ParentOf => "parent",
            Self::ChildOf => "child",
            Self::SiblingOf => "sibling",
            Self::FriendOf => "friend",
            Self::TeacherOf => "teacher",
            Self::StudentOf => "student",
            Self::SpouseOf => "spouse",
            Self::Other(s) => s,
        }
    }
}

/// One stored edge: `a` is `relation` of `b`.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredEdge {
    /// From.
    pub a: EntityId,
    /// The relation, `a`'s side.
    pub relation: Relation,
    /// To.
    pub b: EntityId,
    /// What was said that taught it.
    pub source: String,
    /// 0..1.
    pub confidence: f32,
    /// Unix seconds.
    pub valid_from: f64,
    /// Unix seconds; `None` while current.
    pub valid_to: Option<f64>,
}

/// A current edge as seen from one person: "`other` is my `relation`"
/// read as `me` is `relation` of `other`.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    /// The relation from `me`'s side (`ParentOf` = I am their parent).
    pub relation: Relation,
    /// The other person.
    pub other: EntityId,
    /// Their name.
    pub other_name: String,
    /// Confidence of the stored edge.
    pub confidence: f32,
}

/// Whether two relations between the same pair contradict: different
/// words, except the inverse of each other (which is the same edge).
pub fn conflicts(old: &Relation, new: &Relation) -> bool {
    old != new
}

/// An in-memory temporal graph: what [`crate::tools::InMemoryFacts`] keeps,
/// and the reference the SQLite store is tested against.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    /// Every edge, closed ones included.
    pub edges: Vec<StoredEdge>,
}

impl Graph {
    /// Record `a` is `relation` of `b`. A current edge between the pair
    /// that says the same (either direction) is kept; one that says
    /// something else is closed at `now`. `true` if a new edge was opened.
    pub fn relate(
        &mut self,
        a: &EntityId,
        relation: Relation,
        b: &EntityId,
        source: &str,
        confidence: f32,
        now: f64,
    ) -> bool {
        if a == b {
            return false;
        }
        let mut same = false;
        for e in self.edges.iter_mut().filter(|e| e.valid_to.is_none()) {
            let mine = if &e.a == a && &e.b == b {
                Some(e.relation.clone())
            } else if &e.a == b && &e.b == a {
                Some(e.relation.inverse())
            } else {
                None
            };
            if let Some(r) = mine {
                if conflicts(&r, &relation) {
                    e.valid_to = Some(now);
                } else {
                    same = true;
                }
            }
        }
        if same {
            return false;
        }
        self.edges.push(StoredEdge {
            a: a.clone(),
            relation,
            b: b.clone(),
            source: source.to_owned(),
            confidence,
            valid_from: now,
            valid_to: None,
        });
        true
    }

    /// Current edges of `id`, from its side, oldest first; `name` names the
    /// other person.
    pub fn current(&self, id: &EntityId, name: impl Fn(&EntityId) -> String) -> Vec<Edge> {
        self.edges
            .iter()
            .filter(|e| e.valid_to.is_none())
            .filter_map(|e| {
                if &e.a == id {
                    Some((e.relation.clone(), e.b.clone(), e.confidence))
                } else if &e.b == id {
                    Some((e.relation.inverse(), e.a.clone(), e.confidence))
                } else {
                    None
                }
            })
            .map(|(relation, other, confidence)| Edge {
                relation,
                other_name: name(&other),
                other,
                confidence,
            })
            .collect()
    }

    /// Drop every edge touching `id` (forget me).
    pub fn forget(&mut self, id: &EntityId) {
        self.edges.retain(|e| &e.a != id && &e.b != id);
    }
}

/// What to do with a newly heard fact, given what is already held.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactOp {
    /// New: store it.
    Add,
    /// Replaces the held fact (same subject, new value).
    Update(String),
    /// The person retracted the held fact.
    Delete(String),
    /// Already known (or nothing to retract).
    Noop,
}

/// The subject key of a fact: its first two content words, lower-case,
/// with a leading "I"/"my"/pronoun dropped. "likes cricket" and "likes
/// football" share `likes`; "my favourite colour is blue" and "my
/// favourite colour is red" share `favourite colour`.
pub fn subject_key(fact: &str) -> String {
    let words: Vec<String> = fact
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .filter(|w| {
            !matches!(
                w.as_str(),
                "i" | "my"
                    | "he"
                    | "she"
                    | "they"
                    | "his"
                    | "her"
                    | "their"
                    | "i'm"
                    | "am"
                    | "a"
                    | "an"
                    | "the"
                    | "really"
                    | "also"
                    | "now"
                    | "not"
                    | "don't"
                    | "no"
                    | "longer"
            )
        })
        .collect();
    // "favourite X is Y" -> "favourite x"; "likes Y" -> "likes"; "is Y" ->
    // the whole thing (no safe subject).
    match words.as_slice() {
        [] => String::new(),
        [f, x, ..] if f == "favourite" || f == "favorite" => format!("favourite {x}"),
        [v, ..]
            if matches!(
                v.as_str(),
                "likes"
                    | "like"
                    | "loves"
                    | "love"
                    | "plays"
                    | "play"
                    | "lives"
                    | "live"
                    | "works"
                    | "work"
                    | "studies"
                    | "study"
            ) =>
        {
            v.trim_end_matches('s').to_owned()
        }
        [v, x, ..] if v == "has" || v == "have" => format!("has {x}"),
        _ => words.join(" "),
    }
}

fn norm(fact: &str) -> String {
    fact.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .filter(|w| !matches!(w.as_str(), "i" | "my" | "he" | "she" | "they"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The fact write decision, by rules: `retract` = the person took it back
/// ("I don't like cricket any more").
pub fn fact_op(held: &[String], new: &str, retract: bool) -> FactOp {
    let new_n = norm(new);
    if new_n.is_empty() {
        return FactOp::Noop;
    }
    let key = subject_key(new);
    // A single-valued subject ("favourite colour", "lives") is replaced;
    // "likes" can hold many.
    let single = key.starts_with("favourite ") || key == "live" || key == "work" || key == "study";
    if retract {
        // The held fact with the same wording, else (single-valued) the
        // same subject.
        return held
            .iter()
            .find(|h| norm(h) == new_n || norm(h).contains(&new_n))
            .or_else(|| {
                single
                    .then(|| held.iter().find(|h| subject_key(h) == key))
                    .flatten()
            })
            .map_or(FactOp::Noop, |h| FactOp::Delete(h.clone()));
    }
    if held
        .iter()
        .any(|h| norm(h) == new_n || norm(h).contains(&new_n))
    {
        return FactOp::Noop;
    }
    if single && let Some(h) = held.iter().find(|h| subject_key(h) == key) {
        return FactOp::Update(h.clone());
    }
    // A refinement of a held fact ("likes cricket" -> "likes cricket and
    // football") replaces it.
    if let Some(h) = held.iter().find(|h| {
        let hn = norm(h);
        !hn.is_empty() && new_n.contains(&hn)
    }) {
        return FactOp::Update(h.clone());
    }
    FactOp::Add
}

/// Privacy: whether the bot may talk to `speaker` about `about`. Only
/// about the speaker themselves or someone the speaker is related to in
/// the graph. Never another child's details to a stranger.
pub fn may_discuss(speaker: Option<&EntityId>, about: &EntityId, speaker_edges: &[Edge]) -> bool {
    match speaker {
        Some(s) if s == about => true,
        Some(_) => speaker_edges.iter().any(|e| &e.other == about),
        None => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn id(s: &str) -> EntityId {
        EntityId::new(s)
    }

    #[test]
    fn inverses_are_derived() {
        let mut g = Graph::default();
        assert!(g.relate(
            &id("suresh"),
            Relation::ParentOf,
            &id("rohan"),
            "x",
            0.9,
            1.0
        ));
        let r = g.current(&id("rohan"), |i| i.to_string());
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].relation, Relation::ChildOf);
        assert_eq!(r[0].other, id("suresh"));
        let s = g.current(&id("suresh"), |i| i.to_string());
        assert_eq!(s[0].relation, Relation::ParentOf);
        // The same edge said from the other side is not a second edge.
        assert!(!g.relate(
            &id("rohan"),
            Relation::ChildOf,
            &id("suresh"),
            "y",
            0.9,
            2.0
        ));
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn a_conflicting_relation_closes_the_old_edge() {
        let mut g = Graph::default();
        g.relate(&id("a"), Relation::FriendOf, &id("b"), "x", 0.8, 1.0);
        assert!(g.relate(&id("a"), Relation::SiblingOf, &id("b"), "y", 0.9, 5.0));
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.edges[0].valid_to, Some(5.0));
        let cur = g.current(&id("b"), |i| i.to_string());
        assert_eq!(cur.len(), 1);
        assert_eq!(cur[0].relation, Relation::SiblingOf);
    }

    #[test]
    fn relation_words() {
        assert_eq!(Relation::from_word("my son"), Some(Relation::ChildOf));
        assert_eq!(Relation::from_word("Sister"), Some(Relation::SiblingOf));
        assert_eq!(Relation::from_word("father"), Some(Relation::ParentOf));
        assert_eq!(
            Relation::from_word("cousin"),
            Some(Relation::Other("cousin".into()))
        );
        assert_eq!(Relation::from_word("table"), None);
        assert_eq!(Relation::TeacherOf.inverse(), Relation::StudentOf);
        assert_eq!(Relation::parse("parent_of"), Relation::ParentOf);
    }

    #[test]
    fn privacy_only_self_and_relations() {
        let mut g = Graph::default();
        g.relate(
            &id("suresh"),
            Relation::ParentOf,
            &id("rohan"),
            "x",
            0.9,
            1.0,
        );
        let edges = g.current(&id("suresh"), |i| i.to_string());
        assert!(may_discuss(Some(&id("suresh")), &id("rohan"), &edges));
        assert!(may_discuss(Some(&id("suresh")), &id("suresh"), &edges));
        assert!(!may_discuss(Some(&id("suresh")), &id("priya"), &edges));
        assert!(!may_discuss(None, &id("rohan"), &[]));
    }

    #[test]
    fn fact_ops() {
        let held = vec![
            "likes cricket".to_owned(),
            "favourite colour is blue".to_owned(),
        ];
        assert_eq!(fact_op(&held, "Likes cricket.", false), FactOp::Noop);
        assert_eq!(fact_op(&held, "has a dog called Bruno", false), FactOp::Add);
        assert_eq!(
            fact_op(&held, "my favourite colour is red", false),
            FactOp::Update("favourite colour is blue".into())
        );
        assert_eq!(
            fact_op(&held, "likes cricket and chess", false),
            FactOp::Update("likes cricket".into())
        );
        assert_eq!(
            fact_op(&held, "likes cricket", true),
            FactOp::Delete("likes cricket".into())
        );
        assert_eq!(fact_op(&held, "likes tennis", true), FactOp::Noop);
        assert_eq!(
            fact_op(&held, "favourite colour is green", true),
            FactOp::Delete("favourite colour is blue".into())
        );
        assert_eq!(fact_op(&held, "has a cat", true), FactOp::Noop);
        assert_eq!(fact_op(&held, "likes football", false), FactOp::Add);
    }
}
