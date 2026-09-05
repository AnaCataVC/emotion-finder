import csv
import random
import os
from pathlib import Path

# Ensure reproducibility
random.seed(42)

# Pool of filler prefixes for padding fallback to avoid repetitive single-word echoes
_PADDING_PREFIXES_ES = ["pues", "o sea", "digamos que", "en fin,", "la neta,"]
_PADDING_PREFIXES_EN = ["well,", "I mean,", "you know,", "honestly,", "if I'm honest,"]


def generate_spanish():
    quadrants = {
        'alta_positiva': {
            'act': 'alta', 'val': 'positiva',
            'components': [
                ["tengo ganas de", "quiero", "siento que voy a", "estoy listo para", "me muero por", "estoy"],
                ["saltar", "gritar", "correr", "celebrar", "bailar", "reír a carcajadas", "abrazar a todos",
                 "muy alegre", "muy emocionado", "súper feliz", "muy entusiasmado", "eufórico"],
                ["de alegría", "de emoción", "de felicidad", "de entusiasmo", "con todo el gozo"],
                # Physical sensations & core direct emotions
                ["siento electricidad en el cuerpo", "tengo el corazón latiendo a mil", "siento una energía desbordante",
                 "me tiemblan las manos de la emoción", "siento mariposas en el estómago",
                 "estoy muy alegre", "estoy feliz de verte", "me siento muy feliz", "estoy eufórico",
                 "siento un gran entusiasmo", "estoy radiante y alegre", "súper feliz de la vida"],
                # Colloquial & Dialectal
                ["estoy que no quepo de la alegría", "estoy súper motivado", "tengo las pilas puestas", "estoy a tope",
                 "siento que toco el cielo con las manos", "estoy que salto en una pata de lo feliz que estoy"]
            ]
        },
        'alta_negativa': {
            'act': 'alta', 'val': 'negativa',
            'components': [
                ["siento que", "tengo", "me siento", "estoy", "ando"],
                ["me asfixio", "los puños apretados", "la sangre hirviendo", "a punto de estallar", "con los nervios de punta",
                 "muy enojado", "súper furioso", "con mucha rabia", "con mucho miedo", "muy asustado"],
                ["del pánico", "de la rabia", "del estrés", "de la ansiedad", "del coraje", "de la furia"],
                # Mental states / physical / core direct emotions
                ["mi cabeza es un torbellino de pensamientos", "no puedo dejar de pensar en eso", "siento un nudo en la garganta",
                 "tengo taquicardia", "me suda todo el cuerpo", "siento que me va a dar algo",
                 "estoy muy enojado con mi jefe", "tengo rabia", "siento miedo de salir a la calle",
                 "tengo mucho miedo", "siento furia incontrolable", "estoy aterrado con la noticia",
                 "siento que me muero de rabia", "tengo mucha ansiedad por el examen", "no puedo calmarme de la furia"],
                # Colloquial / Dialectal / Negations
                ["no puedo más del estrés", "estoy que echo chispas", "me hierve la sangre", "no me calienta ni el sol",
                 "estoy a punto de perder la cabeza", "estoy con las emociones a flor de piel",
                 "estoy con la mecha corta y cualquier cosa me hace saltar", "estoy con la pera del susto que tengo"]
            ]
        },
        'baja_negativa': {
            'act': 'baja', 'val': 'negativa',
            'components': [
                ["siento", "tengo", "estoy", "me siento", "me noto"],
                ["el cuerpo pesado", "un vacío enorme", "sin fuerzas", "apagado", "hundido", "sin ganas de nada",
                 "muy triste", "profundamente deprimido", "completamente agotado", "desolado"],
                ["y no tengo ganas de salir", "y todo me da igual", "y no le veo sentido a nada", "y quiero llorar",
                 "y me cuesta levantarme", "y no puedo más"],
                # Mental / Physical / core direct emotions & negations
                ["no tengo energías ni para hablar", "siento una pesadez en el pecho", "me arrastro por los rincones",
                 "mi mente está nublada", "siento una opresión en el pecho",
                 "estoy triste", "estoy muy triste", "estoy agotado de tanto trabajar", "me siento deprimido",
                 "estoy desolado y solo", "me da mucha pena lo que paso", "no me siento bien",
                 "no estoy feliz", "estoy nada contento", "no tengo ganas de nada"],
                # Colloquial / Dialectal / Negations
                ["no tengo fuerzas", "estoy por el piso", "me siento bajoneado", "estoy de capa caída",
                 "no doy más", "estoy hecho polvo", "me da lata todo y solo quiero quedarme encerrado",
                 "estoy con el bajon y sin ganas de levantarme de la cama"]
            ]
        },
        'baja_positiva': {
            'act': 'baja', 'val': 'positiva',
            'components': [
                ["estoy", "me siento", "siento", "ando", "me noto"],
                ["respirando profundo", "muy relajado", "en paz", "tranquilo", "en armonía", "sereno", "muy tranquilo", "muy en paz"],
                ["y mis músculos están sueltos", "y mi mente está en paz", "y estoy disfrutando el silencio", "y todo fluye"],
                # Mental / Physical / core direct emotions
                ["siento una paz total", "tengo el cuerpo ligero", "estoy flotando",
                 "mi respiración es pausada y tranquila", "siento una calma inmensa",
                 "me siento en paz conmigo mismo", "estoy sereno y tranquilo", "estoy muy en paz",
                 "disfrutando de la calma", "todo está tranquilo"],
                # Colloquial / Dialectal
                ["estoy modo zen", "me siento como nuevo", "estoy en mi centro", "estoy en las nubes",
                 "me siento súper chill", "estoy piola y relajado disfrutando el silencio y la calma"]
            ]
        }
    }

    data = []
    for q_name, q_data in quadrants.items():
        comps = q_data['components']
        phrases = set()

        # Combinations
        for c1 in comps[0]:
            for c2 in comps[1]:
                for c3 in comps[2]:
                    phrases.add(f"{c1} {c2} {c3}")

        # Standalone physical / mental
        for p in comps[3]:
            phrases.add(p)
            for c in ["porque sí", "hoy", "ahora mismo", "en este momento"]:
                phrases.add(f"{p} {c}")

        # Standalone colloquial
        for p in comps[4]:
            phrases.add(p)
            for m in ["", " de verdad", " completamente", " absolutamente"]:
                if m:
                    phrases.add(f"{p}{m}")

        # Combinations with intensifiers
        for c4 in comps[3] + comps[4]:
            phrases.add(f"últimamente {c4}")
            phrases.add(f"la verdad es que {c4}")
            phrases.add(f"sinceramente {c4}")

        # Ensure run-to-run determinism by sorting before shuffle
        phrases_list = sorted(phrases)
        random.shuffle(phrases_list)
        selected = list(phrases_list)

        # Pad with prefix pool to reach target count per quadrant
        while len(selected) < 175:
            extra = f"{random.choice(_PADDING_PREFIXES_ES)} {random.choice(phrases_list)}"
            if extra not in selected:
                selected.append(extra)

        for text in selected[:175]:
            data.append({
                'text': text,
                'activation': q_data['act'],
                'valence': q_data['val'],
                'quadrant': q_name
            })

    return data


def generate_english():
    quadrants = {
        'alta_positiva': {
            'act': 'alta', 'val': 'positiva',
            'components': [
                ["I feel like", "I want to", "I am ready to", "I'm so pumped to", "I can't wait to", "I am"],
                ["jump", "scream", "run", "celebrate", "dance", "laugh out loud", "hug everyone",
                 "very happy", "so excited", "extremely joyful", "thrilled"],
                ["out of joy", "from excitement", "with happiness", "from enthusiasm", "with delight"],
                # Physical sensations & core emotions
                ["I feel electricity running through my veins", "my heart is racing with excitement", "I have boundless energy",
                 "my hands are shaking from excitement", "I have butterflies in my stomach",
                 "I am very happy", "I feel so joyful and happy", "I am ecstatic and thrilled",
                 "I feel pure euphoria and delight", "I am super excited and happy"],
                # Colloquial & Dialectal
                ["I'm on cloud nine", "I'm super motivated", "I'm pumped up", "I'm absolutely buzzing",
                 "I feel on top of the world", "I am absolutely buzzing with excitement and joy"]
            ]
        },
        'alta_negativa': {
            'act': 'alta', 'val': 'negativa',
            'components': [
                ["I feel like", "I have", "I am", "I feel", "I keep feeling"],
                ["I'm suffocating", "my fists clenched", "my blood boiling", "about to explode", "on edge",
                 "very angry", "so furious", "very scared", "extremely terrified", "full of rage"],
                ["from panic", "from anger", "from stress", "from anxiety", "from rage", "from terror"],
                # Mental states / physical / core emotions
                ["my head is a whirlwind of thoughts", "I can't stop thinking about it", "I feel a lump in my throat",
                 "my heart is pounding out of my chest", "I'm sweating all over", "I feel like I'm having a panic attack",
                 "I am angry", "I am very angry", "I am scared", "I am very scared", "I am furious",
                 "I feel terrifying panic and anxiety", "I am terrified", "I cannot calm down from this anger"],
                # Colloquial / Dialectal / Negations
                ["I can't take this stress anymore", "I'm fuming", "my blood is boiling", "I'm losing my mind",
                 "I'm about to snap", "I'm proper wound up and at the end of my tether",
                 "my nerves are on edge and I feel completely raw", "I am proper wound up"]
            ]
        },
        'baja_negativa': {
            'act': 'baja', 'val': 'negativa',
            'components': [
                ["I feel", "I have", "I am", "I'm feeling", "I'm currently"],
                ["my body heavy", "a huge emptiness", "no strength", "shut down", "down in the dumps", "no desire to do anything",
                 "very sad", "deeply depressed", "completely exhausted", "so lonely", "miserable"],
                ["and I don't want to go out", "and everything is meaningless", "and I see no point",
                 "and I want to cry", "and I can barely get up", "and I have no energy"],
                # Mental / Physical / core emotions & negations
                ["I have no energy even to speak", "I feel a heaviness in my chest", "I'm dragging myself around",
                 "my mind is cloudy", "I feel tightness in my chest",
                 "I am sad", "I am very sad", "I am feeling depressed", "I feel completely exhausted",
                 "I am exhausted from working so much", "I am not happy", "I do not feel good",
                 "not feeling good at all", "I have no hope left", "feeling so down and gloomy"],
                # Colloquial / Dialectal / Negations
                ["I have no strength left", "I've hit rock bottom", "I'm feeling blue",
                 "I feel like crawling into a hole", "I'm totally drained", "I'm absolutely shattered",
                 "I am feeling down in the dumps today and cannot focus"]
            ]
        },
        'baja_positiva': {
            'act': 'baja', 'val': 'positiva',
            'components': [
                ["I am", "I feel", "I'm feeling", "I keep feeling", "I'm currently"],
                ["breathing deeply", "very relaxed", "at peace", "calm", "in harmony", "serene", "very calm", "very peaceful"],
                ["and my muscles are loose", "and my mind is at peace", "and I'm enjoying the silence", "and everything is flowing smoothly"],
                # Mental / Physical / core emotions
                ["I feel total peace", "my body feels light", "I'm floating", "my breathing is slow and steady", "I feel an immense calmness",
                 "I am very calm", "I feel super relaxed", "I am at peace with myself", "everything is quiet and serene"],
                # Colloquial / Dialectal
                ["I'm in zen mode", "I feel like a new person", "I'm completely centered",
                 "I'm in my happy place", "I feel super chill", "feeling right as rain and totally at ease"]
            ]
        }
    }

    data = []
    for q_name, q_data in quadrants.items():
        comps = q_data['components']
        phrases = set()

        # Combinations
        for c1 in comps[0]:
            for c2 in comps[1]:
                for c3 in comps[2]:
                    phrases.add(f"{c1} {c2} {c3}")

        # Standalone physical / mental
        for p in comps[3]:
            phrases.add(p)
            for c in ["today", "right now", "at this moment"]:
                phrases.add(f"{p} {c}")

        # Standalone colloquial
        for p in comps[4]:
            phrases.add(p)
            for m in ["", " really", " completely", " absolutely"]:
                if m:
                    phrases.add(f"{p}{m}")

        # Combinations with intensifiers
        for c4 in comps[3] + comps[4]:
            phrases.add(f"lately {c4}")
            phrases.add(f"honestly {c4}")

        # Ensure run-to-run determinism by sorting before shuffle
        phrases_list = sorted(phrases)
        random.shuffle(phrases_list)
        selected = list(phrases_list)

        # Pad with prefix pool to reach target count per quadrant
        while len(selected) < 175:
            extra = f"{random.choice(_PADDING_PREFIXES_EN)} {random.choice(phrases_list)}"
            if extra not in selected:
                selected.append(extra)

        for text in selected[:175]:
            data.append({
                'text': text,
                'activation': q_data['act'],
                'valence': q_data['val'],
                'quadrant': q_name
            })

    return data


def write_csv(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'activation', 'valence', 'quadrant'])
        writer.writeheader()
        writer.writerows(data)


if __name__ == '__main__':
    es_data = generate_spanish()
    en_data = generate_english()

    base_dir = Path(__file__).resolve().parent
    es_file = base_dir / "emotions_es_v2.csv"
    en_file = base_dir / "emotions_en_v2.csv"

    write_csv(str(es_file), es_data)
    write_csv(str(en_file), en_data)

    print(f"Generated {len(es_data)} rows in {es_file}")
    print(f"Generated {len(en_data)} rows in {en_file}")
