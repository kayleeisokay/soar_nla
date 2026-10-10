"""In-context-learning tasks: six Function Vectors tasks (Todd et al., 2024) and three task switches.

Prompts are the FV paper's `Q:/A:` format with 10 shots, either as raw text (fmt="raw") or placed in
a single chat user turn and followed by a one-word instruction (fmt="chat"). Seeds pick the
demonstrations with zlib.crc32 (not hash(), which Python salts per process).
"""
from __future__ import annotations

import random
import zlib

N_SHOTS = 10
SHIFT = 3   # target of item i = gold of item (i + SHIFT) mod 10, same task

DEMOS = {
 "antonym": [("hot","cold"),("good","bad"),("fast","slow"),("big","small"),("happy","sad"),
             ("light","dark"),("up","down"),("rich","poor"),("hard","soft"),("open","closed"),
             ("early","late"),("strong","weak"),("long","short"),("wet","dry"),("clean","dirty"),
             ("full","empty"),("loud","quiet"),("young","old"),("true","false"),("win","lose")],
 "capitalize": [("apple","Apple"),("river","River"),("stone","Stone"),("garden","Garden"),("window","Window"),
             ("purple","Purple"),("candle","Candle"),("forest","Forest"),("silver","Silver"),("market","Market"),
             ("pencil","Pencil"),("bridge","Bridge"),("summer","Summer"),("island","Island"),("copper","Copper"),
             ("mirror","Mirror"),("basket","Basket"),("shadow","Shadow"),("velvet","Velvet"),("harbor","Harbor")],
 "country-capital": [("France","Paris"),("Japan","Tokyo"),("Italy","Rome"),("Spain","Madrid"),("Germany","Berlin"),
             ("Russia","Moscow"),("Egypt","Cairo"),("Greece","Athens"),("Cuba","Havana"),("Peru","Lima"),
             ("Norway","Oslo"),("Sweden","Stockholm"),("Austria","Vienna"),("Poland","Warsaw"),("Portugal","Lisbon"),
             ("Ireland","Dublin"),("Denmark","Copenhagen"),("Finland","Helsinki"),("Hungary","Budapest"),("Belgium","Brussels")],
 "english-french": [("dog","chien"),("cat","chat"),("house","maison"),("water","eau"),("bread","pain"),
             ("book","livre"),("tree","arbre"),("milk","lait"),("night","nuit"),("king","roi"),
             ("sun","soleil"),("moon","lune"),("fire","feu"),("bird","oiseau"),("flower","fleur"),
             ("cheese","fromage"),("winter","hiver"),("horse","cheval"),("green","vert")],
 "present-past": [("go","went"),("eat","ate"),("run","ran"),("see","saw"),("take","took"),
             ("give","gave"),("write","wrote"),("drink","drank"),("sing","sang"),("swim","swam"),
             ("break","broke"),("speak","spoke"),("drive","drove"),("choose","chose"),("throw","threw"),
             ("wear","wore"),("steal","stole"),("forget","forgot"),("begin","began"),("ride","rode")],
 "singular-plural": [("cat","cats"),("dog","dogs"),("book","books"),("tree","trees"),("car","cars"),
             ("house","houses"),("chair","chairs"),("bottle","bottles"),("flower","flowers"),("river","rivers"),
             ("window","windows"),("garden","gardens"),("island","islands"),("mirror","mirrors"),("candle","candles"),
             ("basket","baskets"),("pencil","pencils"),("bridge","bridges"),("market","markets"),("shadow","shadows")],
}
EVAL = {
 "antonym": [("cold","hot"),("small","big"),("sad","happy"),("slow","fast"),("dark","light"),
             ("weak","strong"),("dirty","clean"),("dry","wet"),("poor","rich"),("late","early")],
 "capitalize": [("table","Table"),("orange","Orange"),("cloud","Cloud"),("winter","Winter"),("guitar","Guitar"),
             ("planet","Planet"),("cotton","Cotton"),("temple","Temple"),("desert","Desert"),("rocket","Rocket")],
 "country-capital": [("China","Beijing"),("Canada","Ottawa"),("Kenya","Nairobi"),("Chile","Santiago"),("Iran","Tehran"),
             ("Turkey","Ankara"),("Thailand","Bangkok"),("Morocco","Rabat"),("Vietnam","Hanoi"),("Sudan","Khartoum")],
 "english-french": [("mother","mère"),("apple","pomme"),("school","école"),("friend","ami"),("street","rue"),
             ("summer","été"),("church","église"),("window","fenêtre"),("shoe","chaussure"),("garden","jardin")],
 "present-past": [("buy","bought"),("sleep","slept"),("teach","taught"),("catch","caught"),("bring","brought"),
             ("build","built"),("feel","felt"),("sell","sold"),("think","thought"),("lose","lost")],
 "singular-plural": [("horse","horses"),("cloud","clouds"),("stone","stones"),("lamp","lamps"),("door","doors"),
             ("plate","plates"),("ticket","tickets"),("rabbit","rabbits"),("pillow","pillows"),("cabin","cabins")],
}

TASKS = list(DEMOS)


def redirect_items(task: str) -> list[dict]:
    ev = EVAL[task]
    return [{"task": task, "idx": i, "query": q, "gold": g,
             "target": ev[(i + SHIFT) % 10][1], "target_query": ev[(i + SHIFT) % 10][0]}
            for i, (q, g) in enumerate(ev)]


SWITCHES = {
 "french->spanish": {
   "source": "English->French", "target": "English->Spanish",
   "src_demos": [("dog","chien"),("cat","chat"),("water","eau"),("bread","pain"),("book","livre"),
                 ("night","nuit"),("milk","lait"),("cheese","fromage"),("tree","arbre"),("king","roi"),
                 ("sun","soleil"),("moon","lune"),("fire","feu"),("bird","oiseau"),("green","vert"),
                 ("horse","cheval"),("winter","hiver"),("flower","fleur"),("island","ile"),("bridge","pont")],
   "tgt_demos": [("dog","perro"),("cat","gato"),("water","agua"),("bread","pan"),("book","libro"),
                 ("night","noche"),("milk","leche"),("cheese","queso"),("tree","arbol"),("king","rey"),
                 ("sun","sol"),("moon","luna"),("fire","fuego"),("bird","pajaro"),("green","verde"),
                 ("horse","caballo"),("winter","invierno"),("flower","flor"),("island","isla"),("bridge","puente")],
   # dropped 'garden' (jardin == jardin) and 'table' (French gold == English query)
   "eval": [("house","maison","casa"),("woman","femme","mujer"),("door","porte","puerta"),
            ("mother","mere","madre"),("school","ecole","escuela"),("street","rue","calle"),
            ("church","eglise","iglesia"),("shoe","chaussure","zapato"),("city","ville","ciudad"),
            ("friend","ami","amigo")],
 },
 "capital->currency": {
   "source": "Country->Capital", "target": "Country->Currency",
   # demo countries chosen so the currency demos are varied rather than mostly 'euro'
   "src_demos": [("France","Paris"),("Italy","Rome"),("Germany","Berlin"),("Egypt","Cairo"),
                 ("Norway","Oslo"),("Denmark","Copenhagen"),("Hungary","Budapest"),("Peru","Lima"),
                 ("Cuba","Havana"),("Kenya","Nairobi"),("Chile","Santiago"),("Morocco","Rabat"),
                 ("Canada","Ottawa"),("Brazil","Brasilia"),("Iceland","Reykjavik"),("Ukraine","Kyiv"),
                 ("Nigeria","Abuja"),("Ghana","Accra"),("Iran","Tehran")],
   "tgt_demos": [("France","euro"),("Italy","euro"),("Germany","euro"),("Egypt","pound"),
                 ("Norway","krone"),("Denmark","krone"),("Hungary","forint"),("Peru","sol"),
                 ("Cuba","peso"),("Kenya","shilling"),("Chile","peso"),("Morocco","dirham"),
                 ("Canada","dollar"),("Brazil","real"),("Iceland","krona"),("Ukraine","hryvnia"),
                 ("Nigeria","naira"),("Ghana","cedi"),("Iran","rial")],
   # dropped Israel (contested capital) and India ('Delhi' vs 'New Delhi')
   "eval": [("Japan","Tokyo","yen"),("Turkey","Ankara","lira"),("China","Beijing","yuan"),
            ("Russia","Moscow","ruble"),("Mexico","Mexico City","peso"),("Sweden","Stockholm","krona"),
            ("Poland","Warsaw","zloty"),("Switzerland","Bern","franc"),("Thailand","Bangkok","baht"),
            ("Vietnam","Hanoi","dong")],
 },
 "antonym->comparative": {
   "source": "Antonym", "target": "Comparative",
   "src_demos": [("hot","cold"),("good","bad"),("big","small"),("happy","sad"),("light","dark"),
                 ("rich","poor"),("hard","soft"),("early","late"),("long","short"),("wet","dry"),
                 ("clean","dirty"),("full","empty"),("loud","quiet"),("young","old"),("deep","shallow"),
                 ("thick","thin"),("sharp","dull"),("wide","narrow"),("heavy","tiny"),("smooth","rough")],
   "tgt_demos": [("hot","hotter"),("good","better"),("big","bigger"),("happy","happier"),("light","lighter"),
                 ("rich","richer"),("hard","harder"),("early","earlier"),("long","longer"),("wet","wetter"),
                 ("clean","cleaner"),("full","fuller"),("loud","louder"),("young","younger"),("deep","deeper"),
                 ("thick","thicker"),("sharp","sharper"),("wide","wider"),("heavy","heavier"),("smooth","smoother")],
   "eval": [("large","small","larger"),("fast","slow","faster"),("strong","weak","stronger"),
            ("bright","dark","brighter"),("cold","hot","colder"),("soft","hard","softer"),
            ("weak","strong","weaker"),("slow","fast","slower"),("warm","cool","warmer"),
            ("dark","light","darker")],
 },
}

SWITCH_NAMES = list(SWITCHES)

# Hand-written task lexicon: task-description phrases read off real AV output; longest first so
# "capital city" wins over "capital".
TASK_LEXICON = {
 "french->spanish": [("French", "Spanish"), ("France", "Spain"), ("Francais", "Spanish")],
 "capital->currency": [
     ("capital cities", "currencies"), ("capital locations", "currencies"),
     ("capital city", "currency"), ("city name", "currency name"),
     ("capitals", "currencies"), ("capital", "currency"),
     ("cities", "currencies"), ("city", "currency"), ("geography", "economics")],
 "antonym->comparative": [
     ("binary pair", "comparative form"), ("opposites", "comparative forms"),
     ("antonyms", "comparative forms"), ("opposite", "comparative form"),
     ("antonym", "comparative form"), ("contrasting", "comparative"),
     ("contrasts", "comparatives"), ("contrast", "comparison")],
}


def switch_items(name: str) -> list[dict]:
    return [{"task": name, "idx": i, "query": q, "gold": g, "target": t}
            for i, (q, g, t) in enumerate(SWITCHES[name]["eval"])]


def demos_text(pairs: list[tuple[str, str]], key: str, query: str, seed: int) -> str:
    """10-shot block ending in `Q:{query}\nA:`; demo choice depends only on (key, query, seed)."""
    rng = random.Random(zlib.crc32(f"{key}|{query}|{seed}".encode()))
    idx = rng.sample([k for k, (a, _) in enumerate(pairs) if a != query], N_SHOTS)
    return "".join(f"Q:{pairs[k][0]}\nA:{pairs[k][1]}\n\n" for k in idx) + f"Q:{query}\nA:"
