# -*- coding: utf-8 -*-
"""02 — Lexicón de anclaje derivado del protocolo de etiquetado.

Traduce el manual de codificación a tres familias de expresiones
regulares que operacionalizan los requisitos 1-4 del test:

  * ACCION       -> requisito 1 (hecho/medida, ocurrida o anunciada),
                    organizada en tres familias de vocabulario (con paro, presencia
                    física) y C (declarativa/indirecta);
  * PROTAGONISTA -> requisito 2 (colectivo de trabajadores: categorías
                    genéricas y organizaciones gremiales nombradas);
  * OBJETO       -> requisito 4 (demanda vinculada a la mercantilización
                    de la fuerza de trabajo).

El requisito 3 (antagonista identificable) no se modela léxicamente:
en la prensa el antagonista casi siempre queda implícito en el verbo
("reclamaron a la empresa", "al Gobierno") y su ausencia léxica no es
informativa. La decisión forma parte de la especificación protocolar.

Con el lexicón se definen dos reglas protocolares puramente
determinísticas, sin datos etiquetados:

  PROTO_amplia   = ACCION y PROTAGONISTA
  PROTO_estricta = ACCION y PROTAGONISTA y OBJETO

Estas reglas NO son el clasificador final: son (a) el ancla con la que
se interpretan tópicos y clusters aprendidos sin supervisión y (b) una
línea de base contra la cual compararlos.

Salida: data/lexicon/lexicon.json
        data/processed/senales_lexicon.parquet
"""
import json
import re
import unicodedata

import pandas as pd

from config import DATA_LEXICON, DATA_PROCESSED

# --------------------------------------------------------------------------
# Lexicón. Cada entrada es una expresión regular que se aplica sobre el
# texto normalizado (minúsculas, sin tildes). Se usan límites de palabra.
# --------------------------------------------------------------------------
LEXICON = {
    # ------------------------------------------------ acciones tipo A ----
    "ACCION_A": [
        r"\bparos?\b", r"\bparazo\b", r"\bhuelgas?\b",
        r"\bparo (general|nacional|activo|total|docente|de actividades)\b",
        r"\bcese de actividades\b", r"\bparalizaci[oa]n de (actividades|tareas)\b",
        r"\bbrazos ca[i]dos\b", r"\bretenci[oa]n de (servicios|tareas|actividades)\b",
        r"\btrabajo a reglamento\b",
    ],
    # ------------------------------------------------ acciones tipo B ----
    "ACCION_B": [
        # el manual ubica el quite de colaboración entre las acciones B
        r"\bquite de colaboraci[oa]n\b",
        r"\bmovilizaci[oa]n(es)?\b", r"\bse movilizar[oa]n\b", r"\bmovilizar[aá]n?\b",
        r"\bmarchar[oa]n\b", r"\bmarcha(s)? (de|por|contra|hacia)\b",
        r"\basamblea(s)?\b", r"\bconcentraci[oa]n(es)?\b", r"\bse concentrar[oa]n\b",
        r"\bpiquetes?\b", r"\bcortes? de (ruta|calle|avenida|puente|v[i]a)\b",
        r"\bcortar[oa]n (la|el) (ruta|calle|avenida|puente|acceso)\b",
        r"\bacampes?\b", r"\bbloqueos?\b", r"\bbloquear[oa]n\b",
        r"\btomas? de (la |el )?(planta|f[a]brica|empresa|edificio|ministerio|municipalidad)\b",
        r"\bocupar[oa]n (la|el) (planta|f[a]brica|empresa|edificio)\b",
        r"\bolla(s)? populares?\b", r"\babrazo (simb[o]lico|solidario)\b",
        r"\bjornada de (lucha|protesta)\b", r"\bprotesta(s|ron|r[a]n)?\b",
        r"\bmanifestaci[oa]n(es)?\b", r"\bse manifestar[oa]n\b",
        r"\bhuelga de hambre\b", r"\bcarpa (blanca|de protesta)\b",
    ],
    # ------------------------------------------- acciones tipo C ----------
    "ACCION_C": [
        r"\bestado de (alerta|asamblea|movilizaci[oa]n)\b",
        r"\balerta y movilizaci[oa]n\b",
        r"\bamenaza(n|ron|r[a]n)? (de |con )?(un )?paro\b",
        r"\banuncia(n|ron|r[a]n)? (un |el )?paro\b",
        r"\bevalu[a]n (un )?paro\b", r"\bno descartan? (un )?paro\b",
        r"\bmedidas? de fuerza\b", r"\bplan de lucha\b",
        r"\bpetitorio\b", r"\breclamos? (salarial|gremial|laboral)\w*\b",
        r"\breclamar[oa]n\b", r"\bexigi(er)?[oa]n\b",
        r"\brechaz[oa] la (oferta|propuesta) salarial\b",
        r"\bconciliaci[oa]n obligatoria\b",
        r"\bpresentar[oa]n (un )?(amparo|recurso|nota|denuncia)\b",
        r"\bparo anunciado\b", r"\bir[a]n? al paro\b", r"\bamenaza de huelga\b",
    ],
    # ---------------------------------------------------- protagonista ----
    "PROTAGONISTA": [
        r"\btrabajador(es|as)?\b", r"\bobrer[oa]s?\b", r"\bemplead[oa]s\b",
        r"\bgremios?\b", r"\bgremial(es)?\b", r"\bsindicat(o|os|al|ales)\b",
        r"\bdelegad[oa]s\b", r"\bcuerpo de delegados\b", r"\bpersonal (de|del)\b",
        r"\bdocentes?\b", r"\bmaestr[oa]s\b", r"\bchoferes\b", r"\bcolectiveros\b",
        r"\bestatales\b", r"\bmunicipales\b", r"\bjudiciales\b", r"\bbancari[oa]s\b",
        r"\benfermer[oa]s\b", r"\bm[e]dic[oa]s\b", r"\bcamioneros\b",
        r"\bmetal[u]rgic[oa]s\b", r"\baceiteros\b", r"\brecolectores\b",
        r"\bdocentes universitari[oa]s\b", r"\bno docentes\b",
        # organizaciones (centrales, federaciones y gremios frecuentes)
        r"\bcgt\b", r"\bcta\b", r"\bate\b", r"\buta\b", r"\buepc\b",
        r"\bsuoem\b", r"\bsurrbac\b", r"\bsep\b", r"\buts\b", r"\bagepj\b",
        r"\bluz y fuerza\b", r"\buom\b", r"\bsmata\b", r"\buocra\b",
        r"\bctera\b", r"\bsadop\b", r"\badiuc\b", r"\bapross\b",
        r"\bla bancaria\b", r"\bsipos\b", r"\bfestram\b", r"\bfatun\b",
        r"\bunter\b", r"\bamet\b", r"\bctep\b", r"\baoita\b", r"\bsitosplad\b",
        r"\bfoetra\b", r"\bapdfa\b", r"\blaf\b", r"\bsuteba\b",
        r"\bconducci[oa]n gremial\b", r"\bcomisi[oa]n interna\b",
        r"\bfederaci[oa]n de (trabajadores|empleados|docentes)\b",
        r"\bsecretari[oa] general (del?|de la) (gremio|sindicato)\b",
    ],
    # --------------------------------------------------- objeto laboral ----
    "OBJETO": [
        r"\bsalari(o|os|al|ales)\b", r"\bsueldos?\b", r"\bhaberes\b",
        r"\bparitari[oa]s?\b", r"\brecomposici[oa]n salarial\b",
        r"\baumento (salarial|de sueldo)\b", r"\bcl[a]usula gatillo\b",
        r"\bdespidos?\b", r"\bdespedid[oa]s\b", r"\bcesant[i]as?\b",
        r"\bsuspensiones\b", r"\bsuspendid[oa]s\b", r"\bretiros voluntarios\b",
        r"\breincorporaci[oa]n\b", r"\bpases? a planta\b",
        r"\bprecarizaci[oa]n\b", r"\bprecarizad[oa]s\b", r"\btercerizad[oa]s\b",
        r"\bcondiciones (laborales|de trabajo)\b", r"\bconvenio colectivo\b",
        r"\bpuestos de trabajo\b", r"\bfuentes? de trabajo\b",
        r"\bdeuda salarial\b", r"\bpago de (sueldos|salarios|haberes)\b",
        r"\bbono (de fin de a[n]o|salarial)\b", r"\bjornada laboral\b",
        r"\breforma laboral\b", r"\bley antidespidos\b", r"\baguinaldo\b",
        r"\bempleo\b", r"\bdesempleo\b", r"\bdesocupaci[oa]n\b",
    ],
    # ------------------------------------- acciones del antagonista -------
    # No se usan para excluir documentos (regla del rebote), sino como
    # descriptores para el análisis de errores y la discusión.
    "ANTAGONISTA_MEDIDA": [
        r"\block ?out\b", r"\bcierre de (la )?(planta|f[a]brica|empresa)\b",
        r"\bdesalojo\b", r"\breprimi[oa]\b", r"\brepresi[oa]n\b",
    ],
}

RE_LEXICON = {k: re.compile("|".join(v)) for k, v in LEXICON.items()}


def normalizar(texto: str) -> str:
    """minúsculas + remoción de tildes/diacríticos (ñ -> n incluida:
    el lexicón está escrito para esa normalización)."""
    t = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def señales(texto_norm: str) -> dict:
    out = {}
    for k, rx in RE_LEXICON.items():
        m = rx.findall(texto_norm)
        out[f"n_{k}"] = len(m)
        out[f"tiene_{k}"] = bool(m)
    out["tiene_ACCION"] = (out["tiene_ACCION_A"] or out["tiene_ACCION_B"]
                           or out["tiene_ACCION_C"])
    out["n_ACCION"] = out["n_ACCION_A"] + out["n_ACCION_B"] + out["n_ACCION_C"]
    out["proto_amplia"] = out["tiene_ACCION"] and out["tiene_PROTAGONISTA"]
    out["proto_estricta"] = out["proto_amplia"] and out["tiene_OBJETO"]
    # La clasificación es binaria: no se deriva ningún subtipo de acción.
    return out


def main() -> None:
    with open(DATA_LEXICON / "lexicon.json", "w", encoding="utf-8") as f:
        json.dump(LEXICON, f, ensure_ascii=False, indent=2)

    df = pd.read_parquet(DATA_PROCESSED / "corpus_limpio.parquet")
    norm = df["texto_analisis"].map(normalizar)
    sen = pd.DataFrame([señales(t) for t in norm])
    sen.insert(0, "id_doc", df["id_doc"].values)
    sen.to_parquet(DATA_PROCESSED / "senales_lexicon.parquet", index=False)

    n = len(sen)
    print(f"Documentos: {n}")
    for c in ["tiene_ACCION_A", "tiene_ACCION_B", "tiene_ACCION_C",
              "tiene_ACCION", "tiene_PROTAGONISTA", "tiene_OBJETO",
              "proto_amplia", "proto_estricta"]:
        print(f"  {c:22s}: {sen[c].sum():6d}  ({sen[c].mean():6.2%})")


if __name__ == "__main__":
    main()
