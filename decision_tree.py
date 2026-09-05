from typing import Dict, Any, List, Optional

"""
Decision Tree Data Structure for Emotion Finder.
Represents 64 emotions mapped to the 4 quadrants of Russell's Circumplex Model.
"""

DECISION_TREES: Dict[str, Dict[str, Any]] = {
    "alta_positiva": {
        "question_es": "¿Sientes un impulso físico vibrante de moverte, saltar o festejar con entusiasmo?",
        "question_en": "Do you feel a vibrant physical impulse to move, jump, or celebrate with enthusiasm?",
        "yes": {

            "question_es": "¿Sientes que tu corazón late rápido con una fuerza casi incontrolable?",
            "question_en": "Do you feel your heart beating fast with an almost uncontrollable force?",
            "yes": {
                "question_es": "¿Sientes una sensación placentera de que estás perdiendo el control de tu cuerpo?",
                "question_en": "Do you feel a pleasant sensation of losing control of your body?",
                "yes": {
                    "question_es": "¿Es una sensación abrumadora de felicidad máxima que recorre todo tu cuerpo?",
                    "question_en": "Is it an overwhelming sensation of maximum happiness flowing through your body?",
                    "yes": {
                        "emotion_es": "Éxtasis",
                        "emotion_en": "Ecstasy",
                        "emoji": "🤩",
                        "description_es": "Experimentas una felicidad tan intensa que casi trasciende lo físico. Todo tu ser vibra de alegría.",
                        "description_en": "You experience happiness so intense it almost transcends the physical. Your whole being vibrates with joy.",
                    },
                    "no": {
                        "emotion_es": "Euforia",
                        "emotion_en": "Euphoria",
                        "emoji": "🙌",
                        "description_es": "Sientes un subidón de energía positiva inmensa, como si pudieras volar.",
                        "description_en": "You feel a massive rush of positive energy, as if you could fly.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes un calor intenso en el pecho dirigido hacia una persona o actividad específica?",
                    "question_en": "Do you feel intense heat in your chest directed toward a specific person or activity?",
                    "yes": {
                        "emotion_es": "Amor apasionado",
                        "emotion_en": "Passionate Love",
                        "emoji": "❤️‍🔥",
                        "description_es": "Sientes una atracción magnética y un deseo profundo y físico hacia alguien o algo.",
                        "description_en": "You feel a magnetic attraction and a deep, physical desire toward someone or something.",
                    },
                    "no": {
                        "emotion_es": "Pasión",
                        "emotion_en": "Passion",
                        "emoji": "🔥",
                        "description_es": "Hay un fuego en tu interior que te impulsa a entregarte por completo a lo que haces.",
                        "description_en": "There is a fire inside you that drives you to give yourself completely to what you do.",
                    }
                }
            },
            "no": {
                "question_es": "¿Sientes la energía vibrando como un impulso irresistible para actuar ahora mismo?",
                "question_en": "Do you feel the energy vibrating as an irresistible impulse to act right now?",
                "yes": {
                    "question_es": "¿Sientes una expansión en el pecho que te hace creer que puedes lograr cualquier cosa?",
                    "question_en": "Do you feel an expansion in your chest making you believe you can achieve anything?",
                    "yes": {
                        "emotion_es": "Empoderamiento",
                        "emotion_en": "Empowerment",
                        "emoji": "🦸",
                        "description_es": "Sientes fuerza y solidez en tu cuerpo; una confianza física en tus propias capacidades.",
                        "description_en": "You feel strength and solidity in your body; a physical confidence in your own capabilities.",
                    },
                    "no": {
                        "emotion_es": "Determinación",
                        "emotion_en": "Determination",
                        "emoji": "💪",
                        "description_es": "Tus músculos están tensos y listos para la acción; tienes una meta clara y física.",
                        "description_en": "Your muscles are tense and ready for action; you have a clear, physical goal.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes un cosquilleo o 'mariposas' en el estómago por algo inminente?",
                    "question_en": "Do you feel a tingling or 'butterflies' in your stomach about something imminent?",
                    "yes": {
                        "emotion_es": "Excitación",
                        "emotion_en": "Excitement",
                        "emoji": "✨",
                        "description_es": "Una chispa eléctrica recorre tu cuerpo ante la perspectiva de algo nuevo o divertido.",
                        "description_en": "An electric spark runs through your body at the prospect of something new or fun.",
                    },
                    "no": {
                        "emotion_es": "Entusiasmo",
                        "emotion_en": "Enthusiasm",
                        "emoji": "😆",
                        "description_es": "Te sientes vibrante y lleno de ganas de participar o involucrarte.",
                        "description_en": "You feel vibrant and full of desire to participate or get involved.",
                    }
                }
            }
        },
        "no": {
            "question_es": "¿Sientes ligereza en el cuerpo, como si flotaras o respiraras mucho más amplio?",
            "question_en": "Do you feel lightness in your body, as if floating or breathing much more spaciously?",
            "yes": {
                "question_es": "¿Sientes que tus ojos se abren más de lo normal y tu vista se expande?",
                "question_en": "Do you feel your eyes opening wider than usual and your vision expanding?",
                "yes": {
                    "question_es": "¿Sientes que tu cuerpo se queda quieto de repente ante algo más grande que tú?",
                    "question_en": "Do you feel your body suddenly become still in the face of something larger than yourself?",
                    "yes": {
                        "emotion_es": "Asombro",
                        "emotion_en": "Awe",
                        "emoji": "🤯",
                        "description_es": "El mundo parece vasto y tu respiración se pausa momentáneamente al contemplar la maravilla.",
                        "description_en": "The world seems vast and your breath pauses momentarily as you contemplate wonder.",
                    },
                    "no": {
                        "emotion_es": "Inspiración",
                        "emotion_en": "Inspiration",
                        "emoji": "💡",
                        "description_es": "Sientes una corriente de energía clara que ilumina tu mente y cuerpo.",
                        "description_en": "You feel a clear stream of energy illuminating your mind and body.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes cosquilleo en el rostro y la urgencia física de reír?",
                    "question_en": "Do you feel a tickle in your face and the physical urge to laugh?",
                    "yes": {
                        "emotion_es": "Diversión",
                        "emotion_en": "Amusement",
                        "emoji": "😂",
                        "description_es": "Tu pecho salta ligeramente y tu rostro no puede evitar sonreír.",
                        "description_en": "Your chest bounces slightly and your face can't help but smile.",
                    },
                    "no": {
                        "emotion_es": "Alegría",
                        "emotion_en": "Joy",
                        "emoji": "😊",
                        "description_es": "Una ligereza cálida que te hace sentir bien, a gusto y sonriente.",
                        "description_en": "A warm lightness that makes you feel good, at ease, and smiling.",
                    }
                }
            },
            "no": {
                "question_es": "¿Sientes tu postura especialmente erguida y el pecho expandido hacia adelante?",
                "question_en": "Do you feel your posture especially upright and your chest expanded forward?",
                "yes": {
                    "question_es": "¿Sientes calor en la cara y una sensación de altura física sobre el entorno?",
                    "question_en": "Do you feel warmth in your face and a sensation of physical height over your surroundings?",
                    "yes": {
                        "emotion_es": "Orgullo",
                        "emotion_en": "Pride",
                        "emoji": "😌",
                        "description_es": "Te sientes grande físicamente, tus hombros están hacia atrás, reconociendo tu propio valor.",
                        "description_en": "You feel physically large, your shoulders are back, recognizing your own worth.",
                    },
                    "no": {
                        "emotion_es": "Gratitud",
                        "emotion_en": "Gratitude",
                        "emoji": "🙏",
                        "description_es": "Sientes el pecho abierto y receptivo, una calidez vibrante hacia los demás.",
                        "description_en": "You feel your chest open and receptive, a vibrating warmth toward others.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes una leve tensión en el cuello o la mirada fija hacia adelante?",
                    "question_en": "Do you feel a slight tension in your neck or your gaze fixed forward?",
                    "yes": {
                        "emotion_es": "Anticipación",
                        "emotion_en": "Anticipation",
                        "emoji": "👀",
                        "description_es": "Tu cuerpo está alerta pero relajado, esperando que algo ocurra.",
                        "description_en": "Your body is alert but relaxed, waiting for something to happen.",
                    },
                    "no": {
                        "emotion_es": "Esperanza",
                        "emotion_en": "Hope",
                        "emoji": "🌱",
                        "description_es": "Sientes una ligereza en el pecho que contrarresta pesares anteriores.",
                        "description_en": "You feel a lightness in your chest that counteracts previous sorrows.",
                    }
                }
            }
        }
    },
    "alta_negativa": {
        "question_es": "¿Sientes tensión física intensa o calor dirigido hacia afuera (hacia otras personas o tu entorno)?",
        "question_en": "Do you feel intense physical tension or heat directed outward (toward others or your surroundings)?",
        "yes": {

            "question_es": "¿Sientes calor intenso en el rostro, mandíbula apretada o los músculos listos para atacar?",
            "question_en": "Do you feel intense heat in your face, a clenched jaw, or muscles ready to attack?",
            "yes": {
                "question_es": "¿Es un impulso agresivo incontrolable de avanzar y destruir/golpear?",
                "question_en": "Is it an uncontrollable aggressive impulse to move forward and destroy/hit?",
                "yes": {
                    "question_es": "¿Tienes los puños cerrados y los músculos completamente rígidos?",
                    "question_en": "Are your fists clenched and muscles completely rigid?",
                    "yes": {
                        "emotion_es": "Ira",
                        "emotion_en": "Anger",
                        "emoji": "😡",
                        "description_es": "Sientes un fuego destructivo, la sangre hirviendo y necesidad de atacar.",
                        "description_en": "You feel a destructive fire, blood boiling, and a need to attack.",
                    },
                    "no": {
                        "emotion_es": "Hostilidad",
                        "emotion_en": "Hostility",
                        "emoji": "😠",
                        "description_es": "Sientes tensión constante y mirada punzante hacia los demás, a la defensiva.",
                        "description_en": "You feel constant tension and a piercing gaze toward others, on the defensive.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes el calor centrado en el cuello y el pecho por una injusticia moral?",
                    "question_en": "Do you feel the heat centered in your neck and chest due to a moral injustice?",
                    "yes": {
                        "emotion_es": "Indignación",
                        "emotion_en": "Indignation",
                        "emoji": "😤",
                        "description_es": "Tu cuerpo se endereza rígidamente al presenciar o vivir algo profundamente injusto.",
                        "description_en": "Your body stiffens rigidly upon witnessing or experiencing something deeply unfair.",
                    },
                    "no": {
                        "emotion_es": "Resentimiento",
                        "emotion_en": "Resentment",
                        "emoji": "😒",
                        "description_es": "Sientes una quemazón sorda y persistente en el estómago o el pecho recordando ofensas.",
                        "description_en": "You feel a dull, persistent burning in your stomach or chest remembering offenses.",
                    }
                }
            },
            "no": {
                "question_es": "¿Sientes una necesidad física visceral de alejarte o repeler algo desagradable?",
                "question_en": "Do you feel a visceral physical need to pull away or repel something unpleasant?",
                "yes": {

                    "question_es": "¿Sientes el estómago revuelto, náuseas o repulsión física en la garganta?",
                    "question_en": "Do you feel an upset stomach, nausea, or physical repulsion in your throat?",
                    "yes": {
                        "emotion_es": "Asco",
                        "emotion_en": "Disgust",
                        "emoji": "🤢",
                        "description_es": "Tu nariz se arruga físicamente y tu cuerpo quiere retroceder de la fuente.",
                        "description_en": "Your nose physically wrinkles and your body wants to back away from the source.",
                    },
                    "no": {
                        "emotion_es": "Frustración",
                        "emotion_en": "Frustration",
                        "emoji": "😫",
                        "description_es": "Sientes tensión como si chocaras contra una pared, apretando los dientes.",
                        "description_en": "You feel tension as if hitting a wall, clenching your teeth.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes picazón bajo la piel, impaciencia y ganas de moverte espasmódicamente?",
                    "question_en": "Do you feel an itch under your skin, impatience, and an urge to move spasmodically?",
                    "yes": {
                        "emotion_es": "Irritabilidad",
                        "emotion_en": "Irritability",
                        "emoji": "🙄",
                        "description_es": "Cada pequeño sonido o toque te resulta físicamente molesto o punzante.",
                        "description_en": "Every little sound or touch feels physically annoying or piercing to you.",
                    },
                    "no": {
                        "emotion_es": "Envidia",
                        "emotion_en": "Envy",
                        "emoji": "🤫",
                        "description_es": "Sientes un vacío tenso en el estómago y tu mirada se clava fijamente en lo que otros tienen.",
                        "description_en": "You feel a tense void in your stomach and your gaze fixates on what others have.",
                    }
                }
            }
        },
        "no": {
            "question_es": "¿Sientes opresión en el pecho, respiración entrecortada y alerta hacia adentro?",
            "question_en": "Do you feel chest tightness, shortness of breath, and inward alertness?",
            "yes": {
                "question_es": "¿Sientes que el corazón se desboca sin control y te paraliza de terror?",
                "question_en": "Do you feel your heart racing uncontrollably, paralyzing you with terror?",
                "yes": {
                    "question_es": "¿Sientes una urgencia física total de huir para sobrevivir, sudor frío y mareo?",
                    "question_en": "Do you feel a total physical urge to flee to survive, cold sweat, and dizziness?",
                    "yes": {
                        "emotion_es": "Pánico",
                        "emotion_en": "Panic",
                        "emoji": "😱",
                        "description_es": "Tu sistema nervioso se ha disparado al máximo, hiperventilas y sientes pérdida de control.",
                        "description_en": "Your nervous system has maxed out, you're hyperventilating and feel a loss of control.",
                    },
                    "no": {
                        "emotion_es": "Miedo",
                        "emotion_en": "Fear",
                        "emoji": "😨",
                        "description_es": "Tus ojos están muy abiertos, tu cuerpo rígido o temblando, preparándose para el peligro.",
                        "description_en": "Your eyes are wide open, your body rigid or shaking, preparing for danger.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes un nudo constante en el estómago, temblor leve y tensión por el futuro?",
                    "question_en": "Do you feel a constant knot in your stomach, slight trembling, and tension about the future?",
                    "yes": {
                        "emotion_es": "Ansiedad",
                        "emotion_en": "Anxiety",
                        "emoji": "😰",
                        "description_es": "Tu pecho se siente apretado y tu respiración es superficial, como anticipando una amenaza difusa.",
                        "description_en": "Your chest feels tight and your breathing is shallow, anticipating a vague threat.",
                    },
                    "no": {
                        "emotion_es": "Nerviosismo",
                        "emotion_en": "Nervousness",
                        "emoji": "😬",
                        "description_es": "Tienes tics, mueves las manos o pies constantemente, con una agitación superficial.",
                        "description_en": "You have tics, move your hands or feet constantly, with a superficial agitation.",
                    }
                }
            },
            "no": {
                "question_es": "¿Sientes una sobrecarga mental opresiva que te agota o sofoca físicamente?",
                "question_en": "Do you feel an oppressive mental overload that physically exhausts or suffocates you?",
                "yes": {

                    "question_es": "¿Sientes que el entorno se te viene encima o te aplasta físicamente?",
                    "question_en": "Do you feel the environment is closing in on you or physically crushing you?",
                    "yes": {
                        "emotion_es": "Agobio",
                        "emotion_en": "Overwhelm",
                        "emoji": "😵",
                        "description_es": "Tus hombros se hunden por el peso invisible y sientes que apenas puedes respirar por la presión.",
                        "description_en": "Your shoulders sag from the invisible weight and you feel you can barely breathe from the pressure.",
                    },
                    "no": {
                        "emotion_es": "Desesperación",
                        "emotion_en": "Desperation",
                        "emoji": "😩",
                        "description_es": "Sientes un dolor agudo en el pecho, como si te desgarraran, luchando sin salida.",
                        "description_en": "You feel sharp pain in your chest, as if being torn apart, struggling with no way out.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes un hueco o ardor en el estómago combinado con un nudo en la garganta?",
                    "question_en": "Do you feel a hollow or burning sensation in your stomach combined with a lump in your throat?",
                    "yes": {
                        "emotion_es": "Culpa",
                        "emotion_en": "Guilt",
                        "emoji": "😔",
                        "description_es": "Tu postura se encoge hacia adentro, como queriendo esconderte, sintiendo un peso interno.",
                        "description_en": "Your posture shrinks inward, as if wanting to hide, feeling an internal weight.",
                    },
                    "no": {
                        "emotion_es": "Obsesión",
                        "emotion_en": "Obsession",
                        "emoji": "😵‍💫",
                        "description_es": "Sientes la cabeza caliente y tensa, dando vueltas en bucle físico sobre un solo tema.",
                        "description_en": "You feel your head hot and tense, physically looping on a single subject.",
                    }
                }
            }
        }
    },
    "baja_negativa": {
        "question_es": "¿Predomina en tu cuerpo una pesadez física densa y sensación de hundimiento?",
        "question_en": "Does your body predominantly feel a dense physical heaviness and sinking sensation?",
        "yes": {

            "question_es": "¿Sientes un nudo apretado en la garganta y fuerte opresión en el pecho?",
            "question_en": "Do you feel a tight lump in your throat and strong chest tightness?",
            "yes": {
                "question_es": "¿Sientes ganas inminentes de llorar, humedad o debilidad en los ojos?",
                "question_en": "Do you feel imminent urges to cry, moisture, or weakness in your eyes?",
                "yes": {
                    "question_es": "¿Esta sensación de llanto es abrumadora, como un vacío físico en el pecho por una pérdida?",
                    "question_en": "Is this crying sensation overwhelming, like a physical void in your chest from a loss?",
                    "yes": {
                        "emotion_es": "Duelo",
                        "emotion_en": "Grief",
                        "emoji": "🥀",
                        "description_es": "El cuerpo duele físicamente; sientes espasmos de llanto y un vacío desgarrador.",
                        "description_en": "The body physically aches; you feel crying spasms and a tearing emptiness.",
                    },
                    "no": {
                        "emotion_es": "Tristeza",
                        "emotion_en": "Sadness",
                        "emoji": "😢",
                        "description_es": "Tu rostro decae, la mirada se pierde hacia abajo y hay lágrimas contenidas o derramadas.",
                        "description_en": "Your face droops, your gaze is lost downward, and tears are held back or shed.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes que el rostro te arde de repente y quieres hacerte físicamente invisible?",
                    "question_en": "Do you feel your face suddenly burning and you want to make yourself physically invisible?",
                    "yes": {
                        "emotion_es": "Vergüenza",
                        "emotion_en": "Shame",
                        "emoji": "😳",
                        "description_es": "Sientes un calor punzante en la cara y cuello, y encoges los hombros instintivamente.",
                        "description_en": "You feel a piercing heat in your face and neck, and instinctively shrug your shoulders.",
                    },
                    "no": {
                        "emotion_es": "Decepción",
                        "emotion_en": "Disappointment",
                        "emoji": "😞",
                        "description_es": "Sientes cómo tus hombros y energía caen repentinamente, soltando un suspiro pesado.",
                        "description_en": "You feel your shoulders and energy suddenly drop, releasing a heavy sigh.",
                    }
                }
            },
            "no": {
                "question_es": "¿Sientes una falta total de energía muscular para sostener tu propio cuerpo?",
                "question_en": "Do you feel a total lack of muscular energy to hold up your own body?",
                "yes": {
                    "question_es": "¿Sientes que tus brazos y piernas son de plomo y simplemente no responden?",
                    "question_en": "Do you feel your arms and legs are made of lead and simply don't respond?",
                    "yes": {
                        "emotion_es": "Fatiga",
                        "emotion_en": "Fatigue",
                        "emoji": "🥱",
                        "description_es": "El agotamiento es físico en cada músculo y tus párpados caen sin remedio.",
                        "description_en": "The exhaustion is physical in every muscle and your eyelids droop hopelessly.",
                    },
                    "no": {
                        "emotion_es": "Desgana",
                        "emotion_en": "Reluctance",
                        "emoji": "😒",
                        "description_es": "Te arrastras para moverte, haciendo las cosas con lentitud y pesadez palpable.",
                        "description_en": "You drag yourself to move, doing things with palpable slowness and heaviness.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes que tus músculos se rinden, tu postura colapsa hacia abajo soltando la lucha?",
                    "question_en": "Do you feel your muscles surrender, your posture collapses downward giving up the fight?",
                    "yes": {
                        "emotion_es": "Desesperanza",
                        "emotion_en": "Hopelessness",
                        "emoji": "🌑",
                        "description_es": "Sientes un peso muerto en el pecho; tu mirada está vacía y fija sin enfocar nada.",
                        "description_en": "You feel a dead weight in your chest; your gaze is empty and fixed focusing on nothing.",
                    },
                    "no": {
                        "emotion_es": "Resignación",
                        "emotion_en": "Resignation",
                        "emoji": "🫥",
                        "description_es": "Sientes una exhalación profunda, los hombros caen, dejando de resistir físicamente.",
                        "description_en": "You feel a deep exhalation, shoulders drop, ceasing physical resistance.",
                    }
                }
            }
        },
        "no": {
            "question_es": "¿Sientes frío, un hueco en el centro del pecho y embotamiento sensorial?",
            "question_en": "Do you feel cold, a hollow in the center of your chest, and sensory dullness?",
            "yes": {
                "question_es": "¿Sientes una punzada o tirón físico en el pecho por algo o alguien ausente?",
                "question_en": "Do you feel a physical twinge or tug in your chest for something or someone absent?",
                "yes": {

                    "question_es": "¿Este hueco y tirón en el pecho se asocia fuertemente con imágenes del pasado?",
                    "question_en": "Is this hollow and tug in the chest strongly associated with images from the past?",
                    "yes": {
                        "emotion_es": "Nostalgia",
                        "emotion_en": "Nostalgia",
                        "emoji": "📻",
                        "description_es": "Sientes una punzada dulce pero dolorosa en el corazón recordando olores o imágenes.",
                        "description_en": "You feel a sweet but painful twinge in your heart recalling smells or images.",
                    },
                    "no": {
                        "emotion_es": "Melancolía",
                        "emotion_en": "Melancholy",
                        "emoji": "🌧️",
                        "description_es": "Una pesadez difusa y fría que te hace mirar por la ventana con la mirada perdida.",
                        "description_en": "A diffuse, cold heaviness that makes you stare out the window with a lost gaze.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes frío en las extremidades y la sensación física de que nadie te sostiene?",
                    "question_en": "Do you feel cold in your extremities and the physical sensation that no one is holding you?",
                    "yes": {
                        "emotion_es": "Soledad",
                        "emotion_en": "Loneliness",
                        "emoji": "👤",
                        "description_es": "Un frío físico penetrante; sientes tu piel como una barrera que te aísla del mundo.",
                        "description_en": "A piercing physical cold; you feel your skin as a barrier isolating you from the world.",
                    },
                    "no": {
                        "emotion_es": "Desamparo",
                        "emotion_en": "Helplessness",
                        "emoji": "🏚️",
                        "description_es": "Sientes el cuerpo pequeño, encogido, vulnerable y físicamente desprotegido.",
                        "description_en": "You feel your body small, shrunk, vulnerable, and physically unprotected.",
                    }
                }
            },
            "no": {
                "question_es": "¿Sientes que tus sentidos físicos están bloqueados, embotados o sin reaccionar?",
                "question_en": "Do you feel your physical senses are blocked, dulled, or unresponsive?",
                "yes": {
                    "question_es": "¿Sientes una ausencia total y absoluta de sensaciones físicas en tu torso?",
                    "question_en": "Do you feel a total and absolute absence of physical sensations in your torso?",
                    "yes": {
                        "emotion_es": "Vacío",
                        "emotion_en": "Emptiness",
                        "emoji": "🕳️",
                        "description_es": "Te sientes como un caparazón. Físicamente no logras registrar ninguna sensación interior.",
                        "description_en": "You feel like a shell. Physically you fail to register any inner sensation.",
                    },
                    "no": {
                        "emotion_es": "Apatía",
                        "emotion_en": "Apathy",
                        "emoji": "😐",
                        "description_es": "Tus músculos están flojos pero apagados; hay una total falta de respuesta facial.",
                        "description_en": "Your muscles are loose but dull; there is a total lack of facial response.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes inquietud leve y mirada desenfocada, pero sin motivación para moverte?",
                    "question_en": "Do you feel mild restlessness and unfocused gaze, but with no motivation to move?",
                    "yes": {
                        "emotion_es": "Inseguridad",
                        "emotion_en": "Insecurity",
                        "emoji": "🫣",
                        "description_es": "Sientes un ligero temblor en las manos o la voz, y encoges el pecho para protegerte.",
                        "description_en": "You feel a slight tremor in your hands or voice, and shrink your chest to protect yourself.",
                    },
                    "no": {
                        "emotion_es": "Aburrimiento",
                        "emotion_en": "Boredom",
                        "emoji": "🥱",
                        "description_es": "Sientes pesadez en los párpados combinada con suspiros frecuentes y cuerpo escurrido.",
                        "description_en": "You feel heaviness in your eyelids combined with frequent sighs and a slouched body.",
                    }
                }
            }
        }
    },
    "baja_positiva": {
        "question_es": "¿Sientes tu cuerpo principalmente suelto, liviano y libre de cualquier tensión?",
        "question_en": "Does your body feel primarily loose, weightless, and free of any tension?",
        "yes": {

            "question_es": "¿Sientes una ligereza excepcional, como si estuvieras flotando o perdiendo gravedad?",
            "question_en": "Do you feel an exceptional lightness, as if floating or losing gravity?",
            "yes": {
                "question_es": "¿Sientes que tu mente está espaciosa, despejada y tu rostro completamente sin tensión?",
                "question_en": "Do you feel your mind is spacious, clear, and your face completely tension-free?",
                "yes": {
                    "question_es": "¿Sientes un silencio físico profundo en tu interior, un equilibrio perfecto?",
                    "question_en": "Do you feel a deep physical silence inside, a perfect balance?",
                    "yes": {
                        "emotion_es": "Serenidad",
                        "emotion_en": "Serenity",
                        "emoji": "🧘",
                        "description_es": "El aire fluye sin esfuerzo. Tu cuerpo y mente se sienten vastos, claros y en paz cristalina.",
                        "description_en": "Air flows effortlessly. Your body and mind feel vast, clear, and in crystalline peace.",
                    },
                    "no": {
                        "emotion_es": "Paz interior",
                        "emotion_en": "Inner Peace",
                        "emoji": "🕊️",
                        "description_es": "El pecho se siente cálido y abierto, con una respiración rítmica y suave.",
                        "description_en": "The chest feels warm and open, with rhythmic and soft breathing.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes que una carga física enorme acaba de desaparecer de tus hombros?",
                    "question_en": "Do you feel a huge physical burden has just disappeared from your shoulders?",
                    "yes": {
                        "emotion_es": "Alivio",
                        "emotion_en": "Relief",
                        "emoji": "😮‍💨",
                        "description_es": "Tus hombros bajan drásticamente y dejas salir un suspiro largo e involuntario.",
                        "description_en": "Your shoulders drop drastically and you let out a long, involuntary sigh.",
                    },
                    "no": {
                        "emotion_es": "Calma",
                        "emotion_en": "Calm",
                        "emoji": "😌",
                        "description_es": "Sientes los músculos desenredarse y el ritmo cardíaco se ralentiza suavemente.",
                        "description_en": "You feel your muscles untangle and your heart rate slows down gently.",
                    }
                }
            },
            "no": {
                "question_es": "¿Sientes el cuerpo apoyado, pesado agradablemente, cediendo a la gravedad?",
                "question_en": "Do you feel your body supported, pleasantly heavy, yielding to gravity?",
                "yes": {
                    "question_es": "¿Sientes una quietud profunda en todos los músculos y extremidades derretidas?",
                    "question_en": "Do you feel deep stillness in all muscles and melted extremities?",
                    "yes": {
                        "emotion_es": "Relajación",
                        "emotion_en": "Relaxation",
                        "emoji": "🛋️",
                        "description_es": "Tus músculos están casi líquidos. La mandíbula, cuello y espalda han soltado toda resistencia.",
                        "description_en": "Your muscles are almost liquid. Jaw, neck, and back have released all resistance.",
                    },
                    "no": {
                        "emotion_es": "Comodidad",
                        "emotion_en": "Comfort",
                        "emoji": "☕",
                        "description_es": "Sientes una suavidad y temperatura agradable envolviendo tu cuerpo.",
                        "description_en": "You feel a pleasant softness and temperature enveloping your body.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes un enfoque visual suave y sin esfuerzo hacia el entorno o tus pensamientos?",
                    "question_en": "Do you feel a soft, effortless visual focus toward your surroundings or thoughts?",
                    "yes": {
                        "emotion_es": "Contemplación",
                        "emotion_en": "Contemplation",
                        "emoji": "🤔",
                        "description_es": "Tu cuerpo está quieto pero tu mirada observa con atención tranquila y desapegada.",
                        "description_en": "Your body is still but your gaze observes with quiet, detached attention.",
                    },
                    "no": {
                        "emotion_es": "Aceptación",
                        "emotion_en": "Acceptance",
                        "emoji": "👐",
                        "description_es": "Sientes el vientre blando y las palmas abiertas, sin oponer resistencia a la realidad.",
                        "description_en": "You feel a soft belly and open palms, offering no resistance to reality.",
                    }
                }
            }
        },
        "no": {
            "question_es": "¿Sientes una calidez suave y brillante enfocada en el área del corazón o pecho?",
            "question_en": "Do you feel a soft, bright warmth focused in the heart or chest area?",
            "yes": {
                "question_es": "¿Es una calidez que te hace inclinarte físicamente e invita a conectar con otros?",
                "question_en": "Is it a warmth that makes you physically lean in and invites connection with others?",
                "yes": {
                    "question_es": "¿Sientes un impulso suave y afectuoso de cuidar o acariciar a alguien?",
                    "question_en": "Do you feel a gentle, affectionate impulse to care for or comfort someone?",
                    "yes": {

                        "emotion_es": "Ternura",
                        "emotion_en": "Tenderness",
                        "emoji": "🥺",
                        "description_es": "Tus ojos se suavizan y tus manos sienten el impulso de acariciar con suma delicadeza.",
                        "description_en": "Your eyes soften and your hands feel the urge to caress with utmost delicacy.",
                    },
                    "no": {
                        "emotion_es": "Cariño",
                        "emotion_en": "Affection",
                        "emoji": "🥰",
                        "description_es": "Sientes un abrazo cálido interno y una suave sonrisa permanente en el rostro.",
                        "description_en": "You feel a warm internal hug and a constant soft smile on your face.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes el pecho suavemente expandido en aprecio silencioso?",
                    "question_en": "Do you feel your chest softly expanded in silent appreciation?",
                    "yes": {
                        "emotion_es": "Gratitud tranquila",
                        "emotion_en": "Quiet Gratitude",
                        "emoji": "🌻",
                        "description_es": "Una tibieza reconfortante inunda tu pecho reconociendo lo bueno que te rodea.",
                        "description_en": "A comforting warmth floods your chest acknowledging the good around you.",
                    },
                    "no": {
                        "emotion_es": "Confianza",
                        "emotion_en": "Trust",
                        "emoji": "🤝",
                        "description_es": "Tu postura es relajada pero abierta, el vientre desprotegido, sabiendo que estás a salvo.",
                        "description_en": "Your posture is relaxed but open, belly unprotected, knowing you are safe.",
                    }
                }
            },
            "no": {
                "question_es": "¿Es una sensación muy estable y duradera de suficiencia en todo el cuerpo?",
                "question_en": "Is it a very stable and long-lasting sensation of enoughness throughout your body?",
                "yes": {
                    "question_es": "¿Sientes que no te falta absolutamente nada y tu estómago asienta agradablemente?",
                    "question_en": "Do you feel you lack absolutely nothing and your stomach settles pleasantly?",
                    "yes": {
                        "emotion_es": "Plenitud",
                        "emotion_en": "Fulfillment",
                        "emoji": "🌕",
                        "description_es": "Sientes que tu cuerpo es un recipiente completo, sin huecos. Estás totalmente saciado.",
                        "description_en": "You feel your body is a full vessel, with no holes. You are totally sated.",
                    },
                    "no": {
                        "emotion_es": "Satisfacción",
                        "emotion_en": "Contentment",
                        "emoji": "☺️",
                        "description_es": "Tus músculos digieren la experiencia; hay un pequeño asentimiento afirmativo en tu cabeza.",
                        "description_en": "Your muscles digest the experience; there is a small affirmative nod in your head.",
                    }
                },
                "no": {
                    "question_es": "¿Sientes que todas las partes de tu cuerpo vibran unidas y sin fricción?",
                    "question_en": "Do you feel all parts of your body vibrate together and without friction?",
                    "yes": {
                        "emotion_es": "Armonía",
                        "emotion_en": "Harmony",
                        "emoji": "🎶",
                        "description_es": "Tu cuerpo se siente como un instrumento afinado, todo fluye sin enganches ni tensiones.",
                        "description_en": "Your body feels like a tuned instrument, everything flows without hitches or tensions.",
                    },
                    "no": {
                        "emotion_es": "Contento",
                        "emotion_en": "Happiness",
                        "emoji": "🙂",
                        "description_es": "Una sonrisa leve y constante en tus labios, y los ojos relajados brillando un poco.",
                        "description_en": "A slight, constant smile on your lips, and relaxed eyes shining a bit.",
                    }
                }
            }
        }
    }
}


def get_tree(quadrant: str) -> Optional[dict]:
    '''Returns the decision tree root for a given quadrant.
    
    Args:
        quadrant (str): One of 'alta_positiva', 'alta_negativa', 'baja_negativa', 'baja_positiva'.
        
    Returns:
        Optional[dict]: The root node of the decision tree for that quadrant, or None.
    '''
    return DECISION_TREES.get(quadrant)



def get_node(quadrant: str, path: str) -> Optional[dict]:
    '''Navigate to a specific node in a quadrant's tree.
    
    Args:
        quadrant (str): The quadrant name.
        path (str): Dot-separated path like 'yes.no.yes.no'. Empty string means root.
        
    Returns:
        Optional[dict]: The node at the specified path, or None if not found.
    '''
    node = get_tree(quadrant)
    if not node:
        return None
    if not path:
        return node
        
    steps = path.split('.')
    for step in steps:
        if isinstance(node, dict) and step in node:
            node = node[step]
        else:
            return None
    return node


def _extract_emotions(node: dict) -> List[dict]:
    '''Helper to extract all leaf (emotion) nodes from a given tree node.'''
    if not node:
        return []
    
    # If it's a leaf node
    if 'emotion_es' in node:
        return [node]
        
    # If it's a question node, recurse
    emotions = []
    if 'yes' in node:
        emotions.extend(_extract_emotions(node['yes']))
    if 'no' in node:
        emotions.extend(_extract_emotions(node['no']))
        
    return emotions


def get_all_emotions() -> list[dict]:
    '''Returns flat list of all 64 emotions with their metadata.
    
    Returns:
        list[dict]: List of all leaf emotion nodes across all quadrants.
    '''
    all_emotions = []
    for quad_name in DECISION_TREES:
        all_emotions.extend(get_quadrant_emotions(quad_name))
    return all_emotions


def get_quadrant_emotions(quadrant: str) -> list[dict]:
    '''Returns the 16 emotions for a specific quadrant.
    
    Args:
        quadrant (str): The quadrant name.
        
    Returns:
        list[dict]: List of 16 emotion dicts for that quadrant.
    '''
    tree = get_tree(quadrant)
    return _extract_emotions(tree)
