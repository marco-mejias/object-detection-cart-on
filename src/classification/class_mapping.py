"""
Mapeo de clases del dataset combinado Klasson + Freiburg a las
categorías del proyecto.

Categorías del proyecto (8):
- fruta:    productos frescos del grupo Fruit (Klasson)
- verdura:  productos frescos del grupo Vegetables (Klasson)
- brick:    envases de cartón (zumos, leches, yogures bebibles)
- lata:     latas de comida y bebida
- botella:  botellas de plástico/cristal (agua, refrescos, aceite, vinagre)
- caja:     cajas rectangulares (cereales, pasta, arroz, té)
- bolsa:    bolsas (chips, café, harina)
- tarro:    tarros pequeños de cristal (mermelada, miel, salsa, especias)

Cuando lleguemos al dataset Monster propio, las nuevas latas se sumarán
a la categoría 'lata' que ahora ya existe.
"""

PROJECT_CATEGORIES = [
    "fruta", "verdura", "brick", "lata", "botella", "caja", "bolsa", "tarro",
]


KLASSON_TO_CATEGORY = {
    # ---- Klasson: frutas ----
    "Apple":            "fruta",
    "Avocado":          "fruta",
    "Banana":           "fruta",
    "Kiwi":             "fruta",
    "Lemon":            "fruta",
    "Lime":             "fruta",
    "Mango":            "fruta",
    "Melon":            "fruta",
    "Nectarine":        "fruta",
    "Orange":           "fruta",
    "Papaya":           "fruta",
    "Passion-Fruit":    "fruta",
    "Peach":            "fruta",
    "Pear":             "fruta",
    "Pineapple":        "fruta",
    "Plum":             "fruta",
    "Pomegranate":      "fruta",
    "Red-Grapefruit":   "fruta",
    "Satsumas":         "fruta",

    # ---- Klasson: verduras ----
    "Asparagus":         "verdura",
    "Aubergine":         "verdura",
    "Cabbage":           "verdura",
    "Carrots":           "verdura",
    "Cucumber":          "verdura",
    "Garlic":            "verdura",
    "Ginger":            "verdura",
    "Leek":              "verdura",
    "Mushroom":          "verdura",
    "Brown-Cap-Mushroom": "verdura",
    "Onion":             "verdura",
    "Pepper":            "verdura",
    "Potato":            "verdura",
    "Red-Beet":          "verdura",
    "Tomato":            "verdura",
    "Zucchini":          "verdura",

    # ---- Klasson: brick (cartón) ----
    "Juice":            "brick",
    "Milk":             "brick",
    "Oatghurt":         "brick",
    "Oat-Milk":         "brick",
    "Sour-Cream":       "brick",
    "Sour-Milk":        "brick",
    "Soyghurt":         "brick",
    "Soy-Milk":         "brick",
    "Yoghurt":          "brick",
}


# Las clases de Freiburg vienen en MAYÚSCULAS, normalmente son nombres
# planos (CEREAL, COFFEE, PASTA, etc.). Las mapeamos por categoría.
FREIBURG_TO_CATEGORY = {
    # ---- lata ----
    "BEANS":           "lata",
    "CORN":            "lata",
    "FISH":            "lata",
    "TOMATO_SAUCE":    "lata",
    "SODA":            "lata",     # latas de refresco
    "BEER":            "lata",     # latas de cerveza (también hay botellas, va a la mayoritaria)

    # ---- botella ----
    "WATER":           "botella",
    "OIL":             "botella",
    "VINEGAR":         "botella",

    # ---- brick ----
    "JUICE":           "brick",
    "MILK":            "brick",

    # ---- caja ----
    "CEREAL":          "caja",
    "PASTA":           "caja",
    "RICE":            "caja",
    "TEA":             "caja",
    "CAKE":            "caja",     # cajas de tartas/bollería
    "CHOCOLATE":       "caja",     # tabletas en caja

    # ---- bolsa ----
    "CHIPS":           "bolsa",
    "FLOUR":           "bolsa",
    "COFFEE":          "bolsa",    # paquetes de café molido
    "SUGAR":           "bolsa",
    "CANDY":           "bolsa",
    "NUTS":            "bolsa",

    # ---- tarro ----
    "JAM":             "tarro",
    "HONEY":           "tarro",
    "SPICES":          "tarro",
}


# Diccionario combinado: prioridad Klasson, luego Freiburg
KLASSON_TO_CATEGORY_FULL = {**KLASSON_TO_CATEGORY, **FREIBURG_TO_CATEGORY}


def map_klasson_class(class_name: str) -> str:
    """
    Devuelve la categoría del proyecto para una clase del dataset combinado.
    Si la clase no está mapeada, devuelve 'otros'.
    """
    return KLASSON_TO_CATEGORY_FULL.get(class_name, "otros")


def get_unmapped_classes(class_names: list) -> list:
    """Devuelve las clases que no están mapeadas (irán a 'otros')."""
    return [c for c in class_names if c not in KLASSON_TO_CATEGORY_FULL]
