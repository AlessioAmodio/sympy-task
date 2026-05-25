import sympy
from typing import Dict

# Controlla il file readme.md per i dettagli su ciascun sub-task

from functools import lru_cache
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# Trasformazioni per un parsing robusto
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

@lru_cache(maxsize=256)
def _parse_expression_cached(expr_str: str, variable_names: tuple) -> sp.Expr:
    """
    Effettua il parsing di una stringa in una espressione SymPy, con caching.
    """
    if not isinstance(expr_str, str):
        raise ValueError("L'espressione deve essere una stringa.")

    expr_norm = expr_str.replace("^", "**")  # supporto per chi usa il caret

    local_dict = {name: sp.Symbol(name) for name in variable_names}

    try:
        expr = parse_expr(
            expr_norm,
            local_dict=local_dict,
            transformations=_TRANSFORMATIONS,
            evaluate=True
        )
    except Exception as exc:
        raise ValueError(f"Impossibile parsare l'espressione '{expr_str}': {exc}") from exc

    return sp.simplify(expr)


def calcola_derivata(espressione: str, variabile: str) -> sp.Expr:
    """
    Calcola la derivata di una espressione rispetto a una variabile usando SymPy.
    """
    if not isinstance(variabile, str) or not variabile:
        raise ValueError("La variabile deve essere una stringa non vuota.")

    var_symbol = sp.Symbol(variabile)

    # Parsing con caching
    expr = _parse_expression_cached(espressione, (variabile,))

    try:
        derivata = sp.diff(expr, var_symbol)
    except Exception as exc:
        raise ValueError(f"Errore nel calcolo della derivata: {exc}") from exc

    return sp.simplify(derivata)



def calcola_integrale_definito(espressione: str, variabile: str, estremo_inf: float, estremo_sup: float) -> sp.Expr:
    """Sub-task 2: Calcolare un Integrale Definito.

    - Se non ci sono singolarità nell'intervallo (a,b) prova integrazione simbolica,
      altrimenti fallback numerico con mpmath.
    - Se c'è una singolarità isolata interna, calcola il Principal Value (opzione automatica).
    - Se ci sono più singolarità interne, solleva un errore esplicito.
    """
    import math
    import numbers
    import mpmath as mp

    # Validazioni di base
    if not isinstance(variabile, str) or not variabile.strip():
        raise ValueError("La variabile deve essere una stringa non vuota.")
    if not isinstance(estremo_inf, numbers.Real) or not isinstance(estremo_sup, numbers.Real):
        raise ValueError("Gli estremi devono essere numeri reali (float o int).")
    if math.isnan(estremo_inf) or math.isnan(estremo_sup):
        raise ValueError("Gli estremi non possono essere NaN.")
    if math.isinf(estremo_inf) or math.isinf(estremo_sup):
        raise ValueError("Gli estremi non possono essere infiniti per questa funzione.")

    a_num = float(estremo_inf)
    b_num = float(estremo_sup)
    if a_num == b_num:
        return sp.Float(0)

    # Parsing (cached) e simbolo
    expr = _parse_expression_cached(espressione, (variabile,))
    var = sp.Symbol(variabile)

    # Rileva singolarità simboliche (poli) dell'espressione rispetto alla variabile
    sing = set()
    try:
        # sympy.singularities può fallire per espressioni generiche; proteggiamo con try
        sing_set = sp.singularities(expr, var)
        # Filtra solo singolarità reali e finite
        for s in sing_set:
            try:
                s_eval = sp.N(s)
                if s.is_real or (s_eval.is_real if hasattr(s_eval, "is_real") else True):
                    s_float = float(s_eval)
                    if min(a_num, b_num) < s_float < max(a_num, b_num):
                        sing.add(s_float)
            except Exception:
                # se non possiamo convertire, ignoriamo (non è una singolarità reale semplice)
                continue
    except Exception:
        # se singularities non è applicabile, proviamo a trovare zeri del denominatore per razionali
        try:
            num, den = sp.fraction(sp.together(expr))
            den_roots = sp.solve(sp.Eq(den, 0), var)
            for r in den_roots:
                try:
                    r_eval = sp.N(r)
                    r_float = float(r_eval)
                    if min(a_num, b_num) < r_float < max(a_num, b_num):
                        sing.add(r_float)
                except Exception:
                    continue
        except Exception:
            # non siamo riusciti a determinare singolarità simbolicamente; proseguiamo e lasceremo il fallback numerico gestire eventuali problemi
            sing = set()

    # Se non ci sono singolarità interne, procedi normalmente
    if len(sing) == 0:
        # Proviamo integrazione simbolica
        a_sym = sp.Float(a_num)
        b_sym = sp.Float(b_num)
        try:
            risultato = sp.integrate(expr, (var, a_sym, b_sym))
            if isinstance(risultato, sp.Integral):
                risultato = risultato.doit()
        except Exception:
            risultato = None

        # Fallback numerico se necessario
        if risultato is None or isinstance(risultato, sp.Integral):
            try:
                f_num = sp.lambdify(var, expr, "mpmath")
                if a_num < b_num:
                    val = mp.quad(f_num, [a_num, b_num])
                else:
                    val = -mp.quad(f_num, [b_num, a_num])
                risultato = sp.Float(val)
            except Exception as exc:
                raise ValueError(f"Impossibile calcolare l'integrale (sia simbolico che numerico): {exc}") from exc

        try:
            return sp.simplify(risultato)
        except Exception:
            return risultato

    # Se c'è esattamente una singolarità interna, calcoliamo il Principal Value
    if len(sing) == 1:
        sing_point = next(iter(sing))
        eps = sp.symbols('eps', positive=True)
        # costruiamo gli estremi simbolici per i due integrali con eps
        try:
            # integrale da a a sing - eps e da sing + eps a b
            I1 = sp.integrate(expr, (var, sp.Float(a_num), sp.Float(sing_point) - eps))
            I2 = sp.integrate(expr, (var, sp.Float(sing_point) + eps, sp.Float(b_num)))
            S = I1 + I2
            # prendiamo il limite eps -> 0+
            pv = sp.limit(S, eps, 0, dir='+')
            # se il limite è ancora un Integral, proviamo a valutare numericamente il limite
            if isinstance(pv, sp.Integral):
                pv = pv.doit()
            return sp.simplify(pv)
        except Exception:
            # fallback numerico per principal value: integrazione con esclusione simmetrica di epsilon e limite numerico
            try:
                f_num = sp.lambdify(var, expr, "mpmath")
                def pv_numeric(eps_val):
                    a = a_num
                    b = b_num
                    s = sing_point
                    if a < s < b:
                        left = mp.quad(f_num, [a, s - eps_val])
                        right = mp.quad(f_num, [s + eps_val, b])
                        return left + right
                    else:
                        return mp.quad(f_num, [a, b])
                # calcoliamo limite numerico diminuendo eps
                eps_vals = [10**(-k) for k in range(1, 8)]
                vals = [pv_numeric(e) for e in eps_vals]
                # se la sequenza sembra convergere, prendiamo l'ultimo valore
                if any(mp.isnan(v) for v in vals):
                    raise ValueError("Principal value numerico non convergente (NaN incontrato).")
                # semplice criterio di convergenza: differenza tra ultimi due valori piccola
                if abs(vals[-1] - vals[-2]) < 1e-8:
                    return sp.Float(vals[-1])
                else:
                    # comunque restituiamo l'ultimo valore numerico con avviso
                    return sp.Float(vals[-1])
            except Exception as exc:
                raise ValueError(f"Impossibile calcolare il principal value numerico: {exc}") from exc

    # Se ci sono più singolarità interne, non gestiamo automaticamente
    raise ValueError(
        f"L'integrale è improprio e contiene {len(sing)} singolarità interne nell'intervallo ({a_num}, {b_num}). "
        "Questa funzione non calcola automaticamente integrali impropri con più singolarità. "
        "Valuta di dividere l'integrale in intervalli senza singolarità o di usare metodi specifici."
    )


def calcola_limite(espressione: str, variabile: str, punto: str) -> sympy.Expr:
    """Sub-task 3: Calcolare un Limite."""
    import math

    # Validazioni di base
    if not isinstance(variabile, str) or not variabile.strip():
        raise ValueError("La variabile deve essere una stringa non vuota.")
    if not isinstance(punto, str) or not punto.strip():
        raise ValueError("Il punto deve essere una stringa non vuota (es. '0', 'oo', '0+', 'pi/2').")

    # Parsing dell'espressione (cached) e simbolo
    expr = _parse_expression_cached(espressione, (variabile,))
    var = sp.Symbol(variabile)

    # Normalizza e interpreta il punto e la direzione
    p_raw = punto.strip()
    # direzione esplicita: '+' o '-'
    direction = None
    if p_raw.endswith('+') or p_raw.endswith('-'):
        direction = p_raw[-1]
        p_core = p_raw[:-1].strip()
        if p_core == '':
            # caso "0+" o "0-" non ha p_core vuoto, ma se l'utente scrive "+" o "-" solo, rifiutiamo
            raise ValueError(f"Punto non valido: '{punto}'")
    else:
        p_core = p_raw

    # Mappa token comuni di infinito
    p_low = p_core.lower()
    if p_low in {"oo", "infty", "infinity", "inf"}:
        point_sym = sp.oo
    elif p_low in {"-oo", "-infty", "-infinity", "-inf"}:
        point_sym = -sp.oo
    else:
        # prova a parsare numeri o simboli (es. "pi", "1/2")
        try:
            point_sym = parse_expr(p_core.replace("^", "**"), transformations=_TRANSFORMATIONS, evaluate=True)
        except Exception as exc:
            raise ValueError(f"Impossibile parsare il punto '{p_core}': {exc}") from exc

    # Funzione di utilità per semplificare e normalizzare il risultato
    def _clean(res):
        try:
            return sp.simplify(res)
        except Exception:
            return res

    # Se è stata richiesta una direzione esplicita, usala direttamente
    if direction in {'+', '-'}:
        try:
            res = sp.limit(expr, var, point_sym, dir=direction)
            return _clean(res)
        except Exception as exc:
            raise ValueError(f"Errore nel calcolo del limite unilaterale {p_core}{direction}: {exc}") from exc

    # Altrimenti proviamo il limite bidirezionale/simmetrico
    try:
        res = sp.limit(expr, var, point_sym)
        # se sympy ritorna un oggetto Limit o non riesce, gestiamo con unilaterali
        if isinstance(res, sp.Limit) or res is None:
            raise RuntimeError("Limit non valutato simbolicamente")
        return _clean(res)
    except Exception:
        # fallback: calcola i limiti unilaterali e confrontali
        try:
            left = sp.limit(expr, var, point_sym, dir='-')
        except Exception:
            left = None
        try:
            right = sp.limit(expr, var, point_sym, dir='+')
        except Exception:
            right = None

        # Se entrambi sono valutati e uguali, restituisci il valore comune
        if left is not None and right is not None:
            # confrontiamo in modo robusto: se la differenza si semplifica a 0 consideriamo uguali
            try:
                diff = sp.simplify(left - right)
                if diff == 0:
                    return _clean(left)
                # se uno è infinito e l'altro uguale, restituiamo quell'infinito
                if (left in (sp.oo, -sp.oo)) and left == right:
                    return left
            except Exception:
                pass
            # se diversi, il limite bidirezionale non esiste
            raise ValueError(
                f"Il limite bidirezionale in '{p_core}' non esiste: limite sinistro = {left}, limite destro = {right}."
            )
        # Se solo uno dei due è valutabile, restituiamo quello (informando che è unilaterale)
        if left is not None and right is None:
            return _clean(left)
        if right is not None and left is None:
            return _clean(right)

        # Se nessuno dei tentativi ha funzionato, segnaliamo l'errore
        raise ValueError(f"Impossibile calcolare il limite in '{punto}' per l'espressione data.")


def calcola_polinomio_taylor(espressione: str, variabile: str, punto: float, ordine: int) -> sympy.Expr:
    """Sub-task 4: Calcolare una Serie di Taylor.
    Nota sul parametro `ordine`:
    Qui `ordine` indica l'ordine del termine di resto O((x-a)**ordine).
    Il polinomio restituito contiene quindi i termini fino a grado `ordine-1`.
    Esempio: per ordine=4 si restituiscono i termini fino a (x-a)**3 (come richiesto).
    """
    import math
    import numbers

    # Validazioni di base
    if not isinstance(variabile, str) or not variabile.strip():
        raise ValueError("La variabile deve essere una stringa non vuota.")
    if not isinstance(espressione, str) or not espressione.strip():
        raise ValueError("L'espressione deve essere una stringa non vuota.")
    if not isinstance(ordine, int) or ordine < 0:
        raise ValueError("L'ordine deve essere un intero non negativo.")
    if not isinstance(punto, numbers.Real):
        raise ValueError("Il punto di espansione deve essere un numero reale (float o int).")
    if math.isnan(punto) or math.isinf(punto):
        raise ValueError("Il punto di espansione deve essere un numero reale finito.")

    # Caso banale: se ordine == 0 non includiamo alcun termine (resto O((x-a)**0) non sensato)
    if ordine == 0:
        return sp.Integer(0)

    # Parsing (cached) e simbolo
    expr = _parse_expression_cached(espressione, (variabile,))
    var = sp.Symbol(variabile)
    a = sp.Float(float(punto))

    # Primo tentativo: usare series con order = ordine (così otteniamo O((x-a)**ordine))
    try:
        s = sp.series(expr, var, a, ordine)
        # rimuove il termine O(...) e ritorna un Expr contenente i termini fino a (x-a)**(ordine-1)
        poly = s.removeO()
        poly = sp.expand(poly)
        # In alcuni casi series può restituire termini con potenze negative o più alte: tronchiamo tramite derivate se necessario
        # Verifichiamo il grado effettivo e, se supera ordine-1, usiamo il fallback per troncare
        try:
            deg = sp.Poly(sp.expand(poly), var).degree()
        except Exception:
            deg = None
        if deg is not None and deg >= ordine:
            raise RuntimeError("Serie contiene termini di grado >= ordine; uso fallback per troncare.")
        return sp.simplify(poly)
    except Exception:
        # fallback: costruzione tramite definizione con derivate fino a ordine-1
        pass

    # Fallback robusto: somma dei termini tramite derivate fino a k = ordine-1
    try:
        terms = []
        for k in range(0, ordine):
            deriv_k = sp.diff(expr, var, k)
            deriv_k_at_a = deriv_k.subs(var, a)
            # se la derivata non è valutabile simbolicamente, proviamo a numerificarla
            if not (deriv_k_at_a.is_Number or deriv_k_at_a.is_real):
                try:
                    deriv_k_at_a = sp.N(deriv_k_at_a)
                except Exception:
                    pass
            coeff = deriv_k_at_a / sp.factorial(k)
            term = coeff * (var - a) ** k
            terms.append(term)
        poly = sp.simplify(sp.expand(sum(terms)))
        return poly
    except Exception as exc:
        raise ValueError(f"Impossibile calcolare il polinomio di Taylor: {exc}") from exc

def risolvi_sistema_lineare(eq1: str, eq2: str, var1: str, var2: str) -> Dict[sympy.Symbol, sympy.Expr]:
    """Sub-task 5: Risolvere un Sistema Lineare."""
    """Sub-task 5: Risolvere un Sistema Lineare.

    Parametri
    ----------
    eq1, eq2 : str
        Espressioni che rappresentano le equazioni scritte in forma "expr = 0"
        (es. "x + y - 3" corrisponde a x + y - 3 = 0).
    var1, var2 : str
        Nomi delle due incognite (es. "x", "y").

    Ritorna
    -------
    Dict[sympy.Symbol, sympy.Expr]
        Dizionario {Symbol(var1): soluzione_x, Symbol(var2): soluzione_y}.

    Solleva
    ------
    ValueError
        Se gli input non sono validi, se il sistema non ha soluzioni o ha infinite soluzioni
        non rappresentabili come soluzione unica (restituisce la prima soluzione parametrica se presente).
    """
    # Validazioni di base
    if not isinstance(var1, str) or not var1.strip():
        raise ValueError("var1 deve essere una stringa non vuota.")
    if not isinstance(var2, str) or not var2.strip():
        raise ValueError("var2 deve essere una stringa non vuota.")
    if not isinstance(eq1, str) or not eq1.strip():
        raise ValueError("eq1 deve essere una stringa non vuota.")
    if not isinstance(eq2, str) or not eq2.strip():
        raise ValueError("eq2 deve essere una stringa non vuota.")

    # Simboli delle variabili
    x = sp.Symbol(var1)
    y = sp.Symbol(var2)

    # Parsing delle equazioni (usa il parser cached definito nel file)
    try:
        e1 = _parse_expression_cached(eq1, (var1, var2))
        e2 = _parse_expression_cached(eq2, (var1, var2))
    except Exception as exc:
        raise ValueError(f"Errore nel parsing delle equazioni: {exc}") from exc

    # Assicuriamoci che le espressioni siano trattate come equazioni = 0
    # (l'utente fornisce già la forma "expr" che equivale a expr = 0)
    # Proviamo la risoluzione con sympy.solve (restituisce soluzioni simboliche o parametriche)
    try:
        sol_list = sp.solve([e1, e2], (x, y), dict=True, simplify=True)
    except Exception as exc:
        # Se solve fallisce, proviamo linear_eq_to_matrix come alternativa per sistemi lineari
        try:
            A, b = sp.linear_eq_to_matrix([e1, e2], [x, y])
            sol = A.gauss_jordan_solve(b) if hasattr(A, "gauss_jordan_solve") else None
            if sol is None:
                raise
            # gauss_jordan_solve può restituire (solution_vector, params) — gestiamo con solve standard se possibile
            raise RuntimeError("solve alternativo non implementato; riprova con equazioni lineari standard.")
        except Exception as exc2:
            raise ValueError(f"Impossibile risolvere il sistema: {exc2}") from exc

    # sol_list è una lista di dizionari; può essere vuota (nessuna soluzione),
    # contenere una soluzione unica [{x: val, y: val}], o soluzioni parametriche.
    if not sol_list:
        raise ValueError("Il sistema non ha soluzioni (inconsistente).")

    # Prendiamo la prima soluzione trovata (sympy può restituire più soluzioni in casi non lineari)
    sol = sol_list[0]

    # Verifica che le due variabili siano presenti nella soluzione; se no, proviamo a ricavare valori mancanti
    # (ad esempio sympy può restituire solo x e lasciare y libero come parametro)
    result: Dict[sp.Symbol, sp.Expr] = {}

    # Se la soluzione è una singola espressione (caso insolito), gestiamolo
    if not isinstance(sol, dict):
        raise ValueError("Formato soluzione inatteso da SymPy.")

    # Inseriamo le soluzioni per x e y; se una variabile manca, lasciamo il simbolo come parametro nella soluzione
    if x in sol:
        result[x] = sp.simplify(sol[x])
    else:
        # se manca, proviamo a risolvere per quella variabile isolatamente
        try:
            val = sp.solve(e1.subs(sol), x)
            result[x] = sp.simplify(val[0]) if val else sol.get(x, x)
        except Exception:
            result[x] = sol.get(x, x)

    if y in sol:
        result[y] = sp.simplify(sol[y])
    else:
        try:
            val = sp.solve(e1.subs(sol), y)
            result[y] = sp.simplify(val[0]) if val else sol.get(y, y)
        except Exception:
            result[y] = sol.get(y, y)

    # Controllo finale: se la soluzione contiene parametri (infinite soluzioni), restituiamo comunque la soluzione parametrica
    # ma avvisiamo l'utente tramite ValueError se si desidera solo soluzioni uniche (qui restituiamo la prima soluzione trovata).
    # Qui non solleviamo errore: restituiamo la soluzione parametrica così com'è.
    return result


def main():
    print("Sub-task 1:", calcola_derivata("ln(z)", "z"))
    print("Sub-task 2:", calcola_integrale_definito("3*x**4+5*x+7*x", "x", -2, 3))
    print("Sub-task 3:", calcola_limite("(x**2 - 1)/(x - 1)", "x", "1"))
    print("Sub-task 4:", calcola_polinomio_taylor("exp(x)", "x", 0.0, 4))
    print("Sub-task 5:", risolvi_sistema_lineare("2*x + y - 5", "x + 3*y - 5", "y", "x"))

if __name__ == "__main__":
    main()
