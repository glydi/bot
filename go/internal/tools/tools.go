// Package tools gives the model a memory of people.
//
// Every call here runs between turns, never between the user finishing a
// sentence and the first audio coming back. Enrolment in particular is
// deliberately off the fast path: meeting someone new is an ordinary
// conversational turn ("what's your name?") and the gallery write happens while
// the bot is already talking.
package tools

import (
	"fmt"
	"strings"
	"sync"

	"github.com/glydi/bot/go/internal/identity"
	"github.com/glydi/bot/go/internal/llm"
)

// Present reports who the bot currently believes it can see, so enrolment can
// attach a name to the right face. Returns "" when the camera cannot say.
type Present interface {
	// SpeakingPersonID is whoever is talking now, if recognised.
	SpeakingPersonID() string
	// PendingFaceEmbeddings returns embeddings for the face being spoken to,
	// ready to enrol. Empty when there is no usable face.
	PendingFaceEmbeddings() [][]float32
}

type Tools struct {
	store   *identity.Store
	present Present
	mu      sync.Mutex
}

func New(store *identity.Store, present Present) *Tools {
	return &Tools{store: store, present: present}
}

func str(args map[string]any, key string) string {
	if v, ok := args[key]; ok {
		if s, ok := v.(string); ok {
			return strings.TrimSpace(s)
		}
		return strings.TrimSpace(fmt.Sprint(v))
	}
	return ""
}

func ok(fields map[string]any) map[string]any {
	if fields == nil {
		fields = map[string]any{}
	}
	fields["status"] = "ok"
	return fields
}

func fail(reason string) map[string]any {
	return map[string]any{"status": "failed", "reason": reason}
}

// Invoke runs a tool call. It never returns an error: a failed tool is
// something the model should be told about in words so it can recover in
// conversation, not an exception that kills the turn.
func (t *Tools) Invoke(name string, args map[string]any) map[string]any {
	t.mu.Lock()
	defer t.mu.Unlock()

	switch name {
	case "remember_name":
		return t.rememberName(args)
	case "remember_fact":
		return t.rememberFact(args)
	case "recall_person":
		return t.recallPerson(args)
	case "forget_person":
		return t.forgetPerson(args)
	default:
		return fail("unknown tool " + name)
	}
}

func (t *Tools) rememberName(args map[string]any) map[string]any {
	name := str(args, "name")
	if name == "" {
		return fail("a name is required")
	}

	var faces [][]float32
	personID := ""
	if t.present != nil {
		faces = t.present.PendingFaceEmbeddings()
		personID = t.present.SpeakingPersonID()
	}

	if len(faces) == 0 {
		// No camera, or no usable face. Still worth recording the name: the
		// bot can carry it through this conversation, and a voice or face can
		// be bound to the same person later.
		id, err := t.store.EnrolNameOnly(name, personID)
		if err != nil {
			return fail(err.Error())
		}
		return ok(map[string]any{
			"name":      name,
			"person_id": id,
			"note":      "remembered for now; no face captured to recognise them by later",
		})
	}

	id, err := t.store.Enrol(name, personID, identity.Face, faces)
	if err != nil {
		return fail(err.Error())
	}
	return ok(map[string]any{"name": name, "person_id": id})
}

func (t *Tools) rememberFact(args map[string]any) map[string]any {
	name, fact := str(args, "name"), str(args, "fact")
	if fact == "" {
		return fail("nothing to remember")
	}
	person, err := t.resolve(name)
	if err != nil {
		return fail(err.Error())
	}
	if person == nil {
		return fail("I do not know anyone by that name yet")
	}
	if err := t.store.Remember(person.ID, fact); err != nil {
		return fail(err.Error())
	}
	return ok(map[string]any{"name": person.Name})
}

func (t *Tools) recallPerson(args map[string]any) map[string]any {
	name := str(args, "name")
	person, err := t.resolve(name)
	if err != nil {
		return fail(err.Error())
	}
	if person == nil {
		everyone, _ := t.store.Everyone()
		known := make([]string, 0, len(everyone))
		for _, p := range everyone {
			known = append(known, p.Name)
		}
		return map[string]any{
			"status":       "unknown",
			"known_people": known,
		}
	}
	facts := person.Facts
	if facts == nil {
		facts = []string{}
	}
	return ok(map[string]any{"name": person.Name, "facts": facts})
}

func (t *Tools) forgetPerson(args map[string]any) map[string]any {
	person, err := t.resolve(str(args, "name"))
	if err != nil {
		return fail(err.Error())
	}
	if person == nil {
		return fail("I do not know anyone by that name")
	}
	removed, err := t.store.Forget(person.ID)
	if err != nil {
		return fail(err.Error())
	}
	if !removed {
		return fail("nothing was removed")
	}
	return ok(map[string]any{"name": person.Name})
}

// resolve finds a person from a name, falling back to whoever is being spoken
// to when the model omits it ("what do you know about me?").
func (t *Tools) resolve(name string) (*identity.Person, error) {
	if name != "" {
		if p, err := t.store.FindByName(name); err != nil || p != nil {
			return p, err
		}
	}
	if t.present != nil {
		if id := t.present.SpeakingPersonID(); id != "" {
			return t.store.Get(id)
		}
	}
	return nil, nil
}

// Declarations describes the tools to the model.
func (t *Tools) Declarations() []llm.Tool {
	strSchema := func(desc string) llm.Schema {
		return llm.Schema{Type: "string", Desc: desc}
	}
	return []llm.Tool{{FunctionDeclarations: []llm.FunctionDeclaration{
		{
			Name: "remember_name",
			Description: "Attach a name to the person you are currently talking to, " +
				"so you recognise them next time. Call this as soon as someone " +
				"tells you their name, but only if you do not already know them.",
			Parameters: llm.Schema{
				Type:       "object",
				Properties: map[string]llm.Schema{"name": strSchema("The name the person gave you.")},
				Required:   []string{"name"},
			},
		},
		{
			Name: "remember_fact",
			Description: "Store something worth remembering about a person you already " +
				"know -- what they do, what they like, something they asked you to " +
				"keep track of. Do not store things they would not expect you to keep.",
			Parameters: llm.Schema{
				Type: "object",
				Properties: map[string]llm.Schema{
					"name": strSchema("Who the fact is about."),
					"fact": strSchema("One short sentence, in the third person."),
				},
				Required: []string{"name", "fact"},
			},
		},
		{
			Name: "recall_person",
			Description: "Look up what you already know about someone by name. Use this " +
				"when you recognise a person and want to pick the conversation back " +
				"up, or when someone asks what you remember about them.",
			Parameters: llm.Schema{
				Type:       "object",
				Properties: map[string]llm.Schema{"name": strSchema("The person's name.")},
				Required:   []string{"name"},
			},
		},
		{
			Name: "forget_person",
			Description: "Permanently delete a person and every stored face and voice " +
				"sample of them. Call this whenever someone asks you to forget them; " +
				"treat the request as final and confirm once it is done.",
			Parameters: llm.Schema{
				Type:       "object",
				Properties: map[string]llm.Schema{"name": strSchema("The person to forget.")},
				Required:   []string{"name"},
			},
		},
	}}}
}
