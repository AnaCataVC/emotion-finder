import csv
import random
import os

# Ensure reproducibility
random.seed(42)

def generate_spanish():
    quadrants = {
        'alta_positiva': {
            'act': 'alta', 'val': 'positiva',
            'components': [
                ["tengo ganas de", "quiero", "siento que voy a", "estoy listo para", "me muero por"],
                ["saltar", "gritar", "correr", "celebrar", "bailar", "reír a carcajadas", "abrazar a todos"],
                ["de alegría", "de emoción", "de felicidad", "de entusiasmo"],
                # Physical sensations
                ["siento electricidad en el cuerpo", "tengo el corazón latiendo a mil", "siento una energía desbordante", "me tiemblan las manos de la emoción", "siento mariposas en el estómago"],
                # Colloquial
                ["estoy que no quepo de la alegría", "estoy súper motivado", "tengo las pilas puestas", "estoy a tope", "siento que toco el cielo con las manos"]
            ]
        },
        'alta_negativa': {
            'act': 'alta', 'val': 'negativa',
            'components': [
                ["siento que", "tengo", "me siento", "estoy"],
                ["me asfixio", "los puños apretados", "la sangre hirviendo", "a punto de estallar", "con los nervios de punta"],
                ["del pánico", "de la rabia", "del estrés", "de la ansiedad", "del coraje"],
                # Mental states / physical
                ["mi cabeza es un torbellino de pensamientos", "no puedo dejar de pensar en eso", "siento un nudo en la garganta", "tengo taquicardia", "me suda todo el cuerpo", "siento que me va a dar algo"],
                # Colloquial / Negations
                ["no puedo más del estrés", "estoy que echo chispas", "me hierve la sangre", "no me calienta ni el sol", "estoy a punto de perder la cabeza"]
            ]
        },
        'baja_negativa': {
            'act': 'baja', 'val': 'negativa',
            'components': [
                ["siento", "tengo", "estoy", "me siento"],
                ["el cuerpo pesado", "un vacío enorme", "sin fuerzas", "apagado", "hundido", "sin ganas de nada"],
                ["y no tengo ganas de salir", "y todo me da igual", "y no le veo sentido a nada", "y quiero llorar", "y me cuesta levantarme"],
                # Mental / Physical
                ["no tengo energías ni para hablar", "siento una pesadez en el pecho", "me arrastro por los rincones", "mi mente está nublada", "siento una opresión en el pecho"],
                # Colloquial / Negations
                ["no tengo fuerzas", "estoy por el piso", "me siento bajoneado", "estoy de capa caída", "no doy más", "estoy hecho polvo"]
            ]
        },
        'baja_positiva': {
            'act': 'baja', 'val': 'positiva',
            'components': [
                ["estoy", "me siento", "siento"],
                ["respirando profundo", "muy relajado", "en paz", "tranquilo", "en armonía", "sereno"],
                ["y mis músculos están sueltos", "y mi mente está en paz", "y estoy disfrutando el silencio", "y todo fluye"],
                # Mental / Physical
                ["siento una paz total", "tengo el cuerpo ligero", "estoy flotando", "mi respiración es pausada y tranquila", "siento una calma inmensa"],
                # Colloquial
                ["estoy modo zen", "me siento como nuevo", "estoy en mi centro", "estoy en las nubes", "me siento súper chill"]
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
        
        # Combinations of standalone with intensifiers
        for c4 in comps[3] + comps[4]:
            phrases.add(f"últimamente {c4}")
            phrases.add(f"te juro que {c4}")
            phrases.add(f"la verdad es que {c4}")
            phrases.add(f"literalmente {c4}")
            phrases.add(f"sinceramente {c4}")
            
        # Ensure we have at least 130
        phrases_list = list(phrases)
        random.shuffle(phrases_list)
        selected = phrases_list[:max(130, len(phrases_list))]
        
        # If we need more, we duplicate with minor variations
        while len(selected) < 130:
            extra = f"pues {random.choice(phrases_list)}"
            if extra not in selected:
                selected.append(extra)
                
        for text in selected[:150]: # take up to 150
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
                ["I feel like", "I want to", "I am ready to", "I'm so pumped to", "I can't wait to"],
                ["jump", "scream", "run", "celebrate", "dance", "laugh out loud", "hug everyone"],
                ["out of joy", "from excitement", "with happiness", "from enthusiasm"],
                # Physical sensations
                ["I feel electricity running through my veins", "my heart is racing with excitement", "I have boundless energy", "my hands are shaking from excitement", "I have butterflies in my stomach"],
                # Colloquial
                ["I'm on cloud nine", "I'm super motivated", "I'm pumped up", "I'm absolutely buzzing", "I feel on top of the world"]
            ]
        },
        'alta_negativa': {
            'act': 'alta', 'val': 'negativa',
            'components': [
                ["I feel like", "I have", "I am", "I feel"],
                ["I'm suffocating", "my fists clenched", "my blood boiling", "about to explode", "on edge"],
                ["from panic", "from anger", "from stress", "from anxiety", "from rage"],
                # Mental states / physical
                ["my head is a whirlwind of thoughts", "I can't stop thinking about it", "I feel a lump in my throat", "my heart is pounding out of my chest", "I'm sweating all over", "I feel like I'm having a panic attack"],
                # Colloquial / Negations
                ["I can't take this stress anymore", "I'm fuming", "my blood is boiling", "I'm losing my mind", "I'm about to snap"]
            ]
        },
        'baja_negativa': {
            'act': 'baja', 'val': 'negativa',
            'components': [
                ["I feel", "I have", "I am", "I'm feeling"],
                ["my body heavy", "a huge emptiness", "no strength", "shut down", "down in the dumps", "no desire to do anything"],
                ["and I don't want to go out", "and everything is meaningless", "and I see no point", "and I want to cry", "and I can barely get up"],
                # Mental / Physical
                ["I have no energy even to speak", "I feel a heaviness in my chest", "I'm dragging myself around", "my mind is cloudy", "I feel tightness in my chest"],
                # Colloquial / Negations
                ["I have no strength left", "I've hit rock bottom", "I'm feeling blue", "I feel like crawling into a hole", "I'm totally drained", "I'm absolutely shattered"]
            ]
        },
        'baja_positiva': {
            'act': 'baja', 'val': 'positiva',
            'components': [
                ["I am", "I feel", "I'm feeling"],
                ["breathing deeply", "very relaxed", "at peace", "calm", "in harmony", "serene"],
                ["and my muscles are loose", "and my mind is at peace", "and I'm enjoying the silence", "and everything is flowing smoothly"],
                # Mental / Physical
                ["I feel total peace", "my body feels light", "I'm floating", "my breathing is slow and steady", "I feel an immense calmness"],
                # Colloquial
                ["I'm in zen mode", "I feel like a new person", "I'm completely centered", "I'm in my happy place", "I feel super chill"]
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
            for c in ["just because", "today", "right now", "at this moment"]:
                phrases.add(f"{p} {c}")
                
        # Standalone colloquial
        for p in comps[4]:
            phrases.add(p)
            for m in ["", " really", " completely", " absolutely"]:
                if m:
                    phrases.add(f"{p}{m}")
        
        # Combinations of standalone with intensifiers
        for c4 in comps[3] + comps[4]:
            phrases.add(f"lately {c4}")
            phrases.add(f"I swear that {c4}")
            phrases.add(f"the truth is {c4}")
            phrases.add(f"literally {c4}")
            phrases.add(f"honestly {c4}")
            
        # Ensure we have at least 130
        phrases_list = list(phrases)
        random.shuffle(phrases_list)
        selected = phrases_list[:max(130, len(phrases_list))]
        
        # If we need more, we duplicate with minor variations
        while len(selected) < 130:
            extra = f"well, {random.choice(phrases_list)}"
            if extra not in selected:
                selected.append(extra)
                
        for text in selected[:150]: # take up to 150
            data.append({
                'text': text,
                'activation': q_data['act'],
                'valence': q_data['val'],
                'quadrant': q_name
            })
            
    return data

def write_csv(filename, data):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'activation', 'valence', 'quadrant'])
        writer.writeheader()
        writer.writerows(data)

if __name__ == '__main__':
    from pathlib import Path
    es_data = generate_spanish()
    en_data = generate_english()
    
    base_dir = Path(__file__).resolve().parent
    es_file = base_dir / "emotions_es_v2.csv"
    en_file = base_dir / "emotions_en_v2.csv"
    
    write_csv(str(es_file), es_data)
    write_csv(str(en_file), en_data)
    
    print(f"Generated {len(es_data)} rows in {es_file}")
    print(f"Generated {len(en_data)} rows in {en_file}")
