> **Created:** 2026-09-04
> **Last Updated:** 2026-09-04
> **Topic:** Dialectal Idioms & Affective Mapping (Chilean Spanish & British English)

# Dialectal Idioms & Affective Mapping (Russell's Circumplex Model)

## 1. Context & Motivation

Idiomatic expressions of emotional states cannot be translated literally across languages without completely losing their affective semantics. For example:
- *"Estar con las emociones a flor de piel"* translated literally to English (*"to have emotions at flower of skin"*) is incomprehensible nonsense.
- Its real emotional meaning is **extreme affective sensitivity, high reactivity, feeling on edge, or being on the verge of tears or breakdown**.
- In Russell's Circumplex Model of Affect (Arousal × Valence), this state is characterized by:
  - **High Arousal (Alta Activación)**: Agitation, increased heart rate, sensory alertness, physical readiness to react.
  - **Negative/Vulnerable Valence (Negativa)**: Irritability, distress, overwhelm, anxiety, or acute sorrow.
  - **Classification**: **Alta Activación · Negativa** (`alta_negativa`).

To ensure the Emotion Finder app handles real colloquial and dialectal user inputs realistically without overcomplicating the NLP pipeline, we curate representative idioms from **Chilean Spanish (Español Chileno)** and **British English**, mapped by their **functional affective equivalence** rather than lexical equivalence.

---

## 2. Dialectal Mapping Table (Russell's 4 Quadrants)

### A. Alta Activación · Negativa (`alta_negativa`)
*State: High energy, reactive distress, tension, anxiety, anger, overwhelm, vulnerability.*

| Idiom (Chilean Spanish) | Functional Meaning | Equivalent (British English) | Core Emotion Target |
|---|---|---|---|
| *"Estar con las emociones a flor de piel"* | High emotional reactivity, hypersensitive, raw | *"My nerves are on edge"* / *"Feeling raw"* | Agobio / Ansiedad / Irritabilidad |
| *"Estar con la mecha corta"* | Highly irritable, reactive, low tolerance | *"Proper wound up"* / *"Ready to blow my top"* | Irritabilidad / Ira |
| *"Estar chato / chata"* | Saturated, overwhelmed, exhausted by pressure | *"At the end of my tether"* / *"At my wits' end"* | Agobio / Frustración |
| *"Estar con la pera"* | Acute fear, nervous trepidation, scared | *"Bricking it"* / *"Shaking like a leaf"* | Miedo / Nerviosismo |
| *"Estar con el corazón en la mano"* | Intense anxiety, dread, suspense | *"Heart in my mouth"* / *"Tearing my hair out"* | Pánico / Desesperación |
| *"Estar con los cables cruzados"* | Explosive agitation, erratic bad mood | *"Seeing red"* / *"Losing my rag"* | Hostilidad / Ira |

---

### B. Baja Activación · Negativa (`baja_negativa`)
*State: Low energy, withdrawal, despondency, gloom, exhaustion, apathy.*

| Idiom (Chilean Spanish) | Functional Meaning | Equivalent (British English) | Core Emotion Target |
|---|---|---|---|
| *"Estar bajoneado / con el bajón"* | Depressed, melancholic, dispirited | *"Down in the dumps"* / *"Feeling blue"* | Tristeza / Melancolía |
| *"Me da lata todo"* | Complete lack of motivation, apathy | *"Can't be bothered with anything"* | Apatía / Desgana |
| *"Estar achacado / achacada"* | Mournful, brooding, sorrowful | *"Gutted"* / *"Feeling down on myself"* | Duelo / Decepción |
| *"No doy más, estoy botado"* | Heavy physical fatigue, worn out | *"Proper knackered"* / *"Dead on my feet"* | Fatiga / Desamparo |

---

### C. Alta Activación · Positiva (`alta_positiva`)
*State: High energy, excitement, joy, triumph, enthusiasm.*

| Idiom (Chilean Spanish) | Functional Meaning | Equivalent (British English) | Core Emotion Target |
|---|---|---|---|
| *"Estar prendido / prendida"* | High energy, lively excitement | *"Buzzing"* / *"Full of beans"* | Entusiasmo / Euforia |
| *"Estar en llamas"* | Unstoppable motivation, passionate flow | *"On fire"* / *"Having a flyer"* | Pasión / Determinación |
| *"Estar que salto en una pata"* | Thrilled, jumping with happiness | *"Over the moon"* / *"Chuffed to bits"* | Alegría / Excitación |
| *"Estar chocho / chocha"* | Overflowing with pride and affection | *"Proud as punch"* | Orgullo / Gratitud |

---

### D. Baja Activación · Positiva (`baja_positiva`)
*State: Low energy, peacefulness, relaxation, quiet contentment.*

| Idiom (Chilean Spanish) | Functional Meaning | Equivalent (British English) | Core Emotion Target |
|---|---|---|---|
| *"Estar piola / piola y relajado"* | Low key, peaceful, undisturbed | *"Chill as anything"* / *"At ease"* | Calma / Serenidad |
| *"Estar tranqui"* | Quiet contentment, unbothered | *"Right as rain"* / *"Ticking along nicely"* | Relajación / Paz interior |
| *"Livianito de sangre"* | Free of burdens, lighthearted peace | *"Cosy and content"* | Alivio / Satisfacción |

---

## 3. Engineering & Dataset Implementation Decisions

1. **Dual Corpus Integration**:
   - Add these idioms to `data/emotions_es_v2.csv` (Chilean Spanish idioms embedded alongside pan-Hispanic expressions).
   - Add the British equivalents to `data/emotions_en_v2.csv` (British idioms alongside standard English).
2. **Neutralization of Meta-Words**:
   - Ensure generic words like *"emociones"* / *"sentimientos"* / *"emotions"* / *"feelings"* are distributed evenly across all 4 quadrants and neutralized in TF-IDF feature selection so they never trigger a one-sided classification shortcut.
3. **Preservation of Key N-Grams**:
   - With TF-IDF `ngram_range=(1, 2)`, expressions like `"flor de piel"` (`"flor piel"`), `"mecha corta"`, `"on edge"`, `"proper wound"`, `"chuffed to bits"` are captured as bigrams and strong contextual unigrams.
