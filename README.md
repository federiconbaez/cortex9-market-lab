# cortex9-market-lab

Framework de investigación, backtesting y paper trading para mercados cripto, en Python.
Usa solo datos públicos de mercado y **nunca envía órdenes reales**: no lee API keys ni llama
a endpoints de trading.

Crypto research, backtesting and paper-trading framework in Python. It only uses public
market data and **never sends real orders**: no API keys are read and no trading endpoint
is called. **English section below.**

> [!WARNING]
> **ES — Leer antes de usar**
> 1. **No se demostró ningún edge.** Ninguna estrategia de este repo mostró ventaja; en el
>    backtest real de abajo las tres pierden y quedan por debajo del buy & hold.
> 2. **Los resultados de backtest no predicen el rendimiento futuro.** Ni los negativos ni
>    un eventual positivo.
> 3. **Con capital muy chico (por ejemplo USD 10) los costos se comen casi todo.** La
>    página de fees de OKX (okx.com/fees, consultada el 30/10/2026; OKX aclara que varía
>    según la región) lista para usuario regular en spot **maker 0.20 % y taker 0.35 %**; el
>    default de ccxt para OKX es maker 0.10 % / taker 0.15 %. El modelo de este repo usa
>    **10 bps (0.10 %)**, más barato que ambos, o sea que los resultados son optimistas. Con
>    35 bps los fees consumieron entre el 15 % y el 22 % del capital en seis meses (ver
>    `docs/examples/backtest_okx_1h_fee35bps.txt`). Como son proporcionales, con USD 10
>    pasa lo mismo, y los costos fijos (retiros, depósitos) pesan todavía más. Los mínimos
>    de orden que publica la API de OKX son **0.00001 BTC** en BTC-USDT (≈ USD 0.78 a
>    78 306) y **0.0001 ETH** en ETH-USDT (≈ USD 0.22 a 2 245); el simulador **no** los
>    aplica.
> 4. **Esta versión no tiene ningún modo de órdenes reales.** No maneja API keys, no usa
>    endpoints privados y no existe ningún camino de código que pueda colocar una orden:
>    el cliente solo puede llamar `load_markets`, `fetch_ohlcv` y `fetch_ticker`
>    (lista blanca en `data/exchange.py`), la instancia de ccxt queda privada dentro de un
>    envoltorio que rechaza cualquier otro atributo, y `tests/test_no_real_orders.py` lo
>    verifica (incluye pruebas de mutación con camelCase, `getattr` con strings
>    concatenados, `eval`/`exec`/`compile`, `__dict__` y acceso al slot privado).
>    **Ojo:** ese test detecta patrones accidentales u obvios; **no** protege contra
>    ofuscación deliberada (alguien que quiera saltearlo puede hacerlo). La barrera real
>    es otra: el código no tiene ni lee credenciales y solo llama endpoints públicos, así
>    que aunque se agregara una llamada de trading, el exchange la rechazaría por falta de
>    autenticación.
>
> **EN — Read before use**
> 1. **No edge has been demonstrated.** In the real backtest below all three strategies
>    lose money and trail buy & hold.
> 2. **Backtest results do not predict future performance**, negative or positive.
> 3. **With very small capital (e.g. USD 10) costs eat most of it.** OKX's fee page
>    (okx.com/fees, read on 2026-10-30; OKX notes fees vary by region) lists regular-user
>    spot **maker 0.20 % / taker 0.35 %**; ccxt's OKX default is maker 0.10 % / taker
>    0.15 %. This repo's model uses **10 bps (0.10 %)**, cheaper than both, so results are
>    optimistic. At 35 bps fees consumed 15–22 % of capital over six months
>    (`docs/examples/backtest_okx_1h_fee35bps.txt`); fees are proportional, so USD 10 fares
>    the same, and fixed costs (withdrawals, deposits) weigh even more. OKX's public API
>    lists minimum order sizes of **0.00001 BTC** (BTC-USDT, ≈ USD 0.78 at 78,306) and
>    **0.0001 ETH** (ETH-USDT, ≈ USD 0.22 at 2,245); the simulator does **not** enforce them.
> 4. **This version has no real-order mode at all**: no API-key handling, no private
>    endpoints, no code path that can place an order. The client can only call
>    `load_markets`, `fetch_ohlcv` and `fetch_ticker` (whitelist in `data/exchange.py`), the
>    ccxt instance is kept private inside a wrapper that refuses any other attribute, and
>    `tests/test_no_real_orders.py` checks it (including mutation tests with camelCase
>    names, string-concatenated `getattr`, `eval`/`exec`/`compile`, `__dict__` and access
>    to the private slot). **Note:** that test catches accidental or obvious patterns; it
>    does **not** defend against deliberate obfuscation (someone determined to bypass it
>    can). The real barrier is that the code holds and reads no credentials and only calls
>    public endpoints, so even if a trading call were added the exchange would reject it as
>    unauthenticated.

> **ES** primero · **EN** abajo

---

## ES

### Aviso importante

- **No se afirma ni se garantiza ninguna ventaja (edge) rentable.** Las tres estrategias
  incluidas son ejemplos didácticos estándar; en el backtest real documentado abajo **las
  tres pierden plata** frente a un buy & hold que ganó en el mismo período.
- Un backtest positivo tampoco probaría nada: hay sobreajuste, sesgo de selección y
  cambios de régimen. Nada de esto es asesoramiento financiero.
- El modo paper es una simulación. No hay integración con órdenes reales y agregarla
  queda fuera del alcance de este repo.

### Qué hace

| Módulo | Qué contiene |
|---|---|
| `cortex9_market_lab.data` | Cliente ccxt de datos públicos (OHLCV, ticker) con reintentos y backoff exponencial con jitter, reconexión (reconstruye la instancia ccxt después de un error transitorio) y fallback a otro exchange (por defecto OKX → Binance → Kraken). Paginación de históricos, caché SQLite, feed de polling que solo expone velas **cerradas**, y generador sintético determinista (GBM y GBM con cambio de régimen). |
| `cortex9_market_lab.strategies` | Interfaz abstracta `Strategy` + registro enchufable (`@register`). Incluye `mean_reversion` (z-score / bandas tipo Bollinger), `momentum` (cruce de EMAs) y `breakout` (canal de Donchian con margen ATR). |
| `cortex9_market_lab.backtest` | Motor event-driven determinista: la señal de la vela *t* se ejecuta en la apertura de *t+1*; stops intra-vela con high/low; métricas (retorno total, CAGR, Sharpe y Sortino anualizados, max drawdown, win rate, profit factor, exposición, cantidad de trades, fees). |
| `cortex9_market_lab.execution` | Modelo de costos (fee y slippage en bps), portafolio simulado (long/short, PnL neto de fees) y router compartido por backtest y paper, así **los dos usan exactamente la misma lógica**. |
| `cortex9_market_lab.risk` | Sizing fijo-fraccional o por volatilidad objetivo, stop-loss / take-profit (gap → se llena en la apertura; si se tocan ambos en la misma vela se asume el stop), tope de exposición por símbolo y total, kill-switch por drawdown máximo y bloqueo de re-entrada después de un stop. |
| `cortex9_market_lab.paper` | Loop de paper trading sobre datos en vivo (polling REST), estado persistido en JSON con escritura atómica, reanudación si la config coincide (fingerprint), cierre ordenado con SIGINT/SIGTERM, duración acotada con `--duration`, corte tras N errores consecutivos. |
| `cortex9_market_lab.observability` | Logging JSON estructurado (una línea por evento) y contadores/gauges serializables. |
| `cortex9_market_lab.config` | Configuración externa en YAML (`configs/*.yaml`) validada con pydantic (claves desconocidas = error). |
| `cortex9_market_lab.cli` | `market-lab strategies | fetch | backtest | paper`. |

### Qué NO hace

- No envía órdenes, no maneja API keys, no toca cuentas.
- No modela libro de órdenes, impacto de mercado, funding, préstamos para short ni
  liquidaciones; el slippage es un porcentaje fijo.
- No usa websockets: el feed en vivo es polling REST (suficiente para velas de 1 minuto o
  más; no sirve para baja latencia).
- No hace optimización de parámetros ni walk-forward; los parámetros de las configs son
  valores por defecto razonables, no ajustados.
- El short está implementado en el simulador pero viene apagado (`allow_short: false`),
  porque en spot real requiere margen.

### Instalación

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Requiere Python ≥ 3.10. Probado localmente con Python 3.12 (y el CI corre 3.10 y 3.12).

### Cómo ejecutar

```sh
# estrategias registradas
market-lab strategies

# bajar históricos públicos a la caché SQLite (data/ohlcv.sqlite)
market-lab fetch --config configs/backtest_okx_1h.yaml

# backtest de las 3 estrategias (salida en runs/backtest_okx_1h/)
market-lab backtest --config configs/backtest_okx_1h.yaml
market-lab backtest --config configs/backtest_okx_1h.yaml --strategy momentum
market-lab backtest --config configs/backtest_okx_1h.yaml --fee-bps 35 --out runs/fee35

# backtest 100 % offline sobre datos sintéticos
market-lab backtest --config configs/synthetic.yaml

# paper trading en vivo durante 150 s (estado en state/paper_okx_1m.json)
market-lab paper --config configs/paper_okx_1m.yaml --strategy momentum --duration 150
# volver a correr el mismo comando reanuda desde el estado guardado
```

Cada backtest escribe `summary.json` y, por estrategia, `report.json`, `trades.csv` y
`equity.csv`. Los logs JSON van a stderr (y a archivo si `logging.file` está seteado).

### Tests y lint

```sh
ruff check . && ruff format --check .
pytest
```

Los tests no usan red: ccxt se reemplaza por un exchange falso. Cubren las fórmulas de las
métricas contra valores calculados a mano, la ausencia de lookahead (truncar la serie o
alterar el futuro no cambia el pasado, tanto en estrategias como en el motor), la
contabilidad de fees y slippage, los límites de riesgo y el kill-switch, las señales sobre
datos sintéticos, la validación de config, el round-trip del estado y el reintento con
backoff, la reconexión y el fallback con un exchange simulado.

### Ejemplo real documentado

Todo lo que sigue es salida literal de corridas reales del 30/10/2026 (UTC), hechas con esta versión del código. Los archivos
crudos están en [`docs/examples/`](docs/examples/).

**Backtest** — fuente `okx:public-ohlcv`, BTC/USDT y ETH/USDT, velas de 1 h, desde
2026-05-01 00:00 UTC hasta 2026-10-28 23:00 UTC (4344 velas por símbolo). Capital inicial
10 000, fee 10 bps, slippage 5 bps, sizing fijo 40 % por entrada (tope 50 % por símbolo,
100 % total), stop-loss 5 %, kill-switch al 25 % de drawdown, solo long.
([`backtest_okx_1h.txt`](docs/examples/backtest_okx_1h.txt))

| metric | mean_reversion | momentum | breakout |
|---|---:|---:|---:|
| total_return | -2.46% | -9.44% | -14.87% |
| cagr | -4.91% | -18.15% | -27.74% |
| sharpe | -0.241 | -0.806 | -1.615 |
| sortino | -0.341 | -1.220 | -2.461 |
| max_drawdown | -8.92% | -16.15% | -22.07% |
| win_rate | 60.00% | 16.67% | 23.44% |
| profit_factor | 0.901 | 0.754 | 0.612 |
| exposure | 34.32% | 52.56% | 41.23% |
| n_trades | 85 | 66 | 64 |
| fees_paid | 668.746 | 500.478 | 477.844 |
| end_equity | 9753.825 | 9055.512 | 8513.235 |

buy & hold (no costs, reference only): BTC/USDT=6.39%, ETH/USDT=4.40%, equal_weight=5.40%

Lectura honesta: en este período y con estos parámetros ninguna estrategia le ganó al
buy & hold, y las tres terminaron en negativo. Los fees solos (≈ 4.8 %–6.7 % del capital)
explican una parte grande de la pérdida. No hubo kill-switch en ninguna. En
[`backtest_synthetic.txt`](docs/examples/backtest_synthetic.txt) está la misma corrida sobre
datos sintéticos (claramente marcados como tales); ahí el kill-switch del 25 % sí se
disparó en `mean_reversion`.

**Sensibilidad a fees** — misma corrida con `--fee-bps 15` (default de ccxt para OKX) y
`--fee-bps 35` (taker de usuario regular según okx.com/fees):

| fee | mean_reversion | momentum | breakout |
|---|---:|---:|---:|
| 10 bps | -2.46% | -9.44% | -14.87% |
| 15 bps | -5.72% | -11.81% | -17.02% |
| 35 bps | -17.74% | -20.67% | -24.13% |

(retorno total; con 35 bps el kill-switch del 25 % se disparó en `breakout`.
[`backtest_okx_1h_fee15bps.txt`](docs/examples/backtest_okx_1h_fee15bps.txt),
[`backtest_okx_1h_fee35bps.txt`](docs/examples/backtest_okx_1h_fee35bps.txt))

**Paper trading** — OKX en vivo, BTC/USDT y ETH/USDT, velas de 1 min, estrategia
`momentum` (EMA 5/20), sizing por volatilidad objetivo (30 % anual, tope 50 % por símbolo y
80 % total), stop 1 %, take-profit 2 %. Corrida de 150 s con polling cada 15 s
([`paper_okx_1m_summary.json`](docs/examples/paper_okx_1m_summary.json),
[`paper_okx_1m.log.jsonl`](docs/examples/paper_okx_1m.log.jsonl)):

- 10 iteraciones, `stop_reason: duration`, 0 errores ni reintentos.
- A las 04:54:29 UTC abrió long simulado en BTC/USDT (0.06379 @ 78 385.27) y ETH/USDT
  (1.33325 @ 2 247.52), precios ya con slippage; exposición total ≈ 80 % (el tope).
- A las 04:56:04 UTC la señal de BTC/USDT se dio vuelta y cerró ese trade con PnL −16.51.
- Al terminar: equity 10 000 → 9 976.66, fees 12.99, 1 trade cerrado, ETH/USDT abierto.

Después, una segunda corrida de 45 s con el mismo comando reanudó el estado
(`"resumed": true`, iteraciones 10 → 13). A las 04:57:02 UTC cerró ETH/USDT por señal con
PnL −12.94. Final: equity 9 970.55, fees acumulados 15.98, 2 trades cerrados, sin
posiciones abiertas:
[`paper_okx_1m_resume_summary.json`](docs/examples/paper_okx_1m_resume_summary.json),
[`paper_state_after_resume.json`](docs/examples/paper_state_after_resume.json).
Tres minutos de paper trading no dicen nada sobre la calidad de una estrategia: esto solo
demuestra que el loop, el riesgo, la persistencia y la reanudación funcionan.

### Limitaciones y riesgos conocidos

- **Sin edge.** Ver arriba. Usá esto para investigar, no para decidir inversiones.
- **Costos simplificados.** Fee y slippage proporcionales y constantes; no hay spread
  variable, profundidad de libro ni funding. El fee por defecto (10 bps) es menor que el
  publicado por OKX para usuario regular.
- **Sin mínimos de orden.** El simulador no aplica `minSz` ni `lotSz` del exchange.
- **Ejecución a la apertura siguiente.** Supone que se puede operar al open de la vela
  siguiente; en velas grandes o mercados ilíquidos el fill real puede ser peor.
- **Stops intra-vela.** Con datos OHLC no se sabe qué tocó primero el precio; se asume el
  stop (conservador). En el backtest un stop se llena **exactamente en el nivel del stop**
  (más slippage); solo si la vela abre más allá del stop (gap) se llena en la apertura. En
  la realidad una orden stop puede llenarse peor. En paper los stops se evalúan solo en
  cada poll, contra el último precio.
- **Paper no modela el spread.** Las ejecuciones simuladas usan el último precio operado
  más el slippage fijo; no se cruza el bid/ask ni se mira la profundidad del libro.
- **Entradas bloqueadas.** Si los topes de exposición bloquean una entrada, se reintenta en
  las velas siguientes mientras la señal siga activa, con el margen que haya.
- **Calidad de datos.** Se descartan filas inválidas (high < low, precio ≤ 0) y se alinean
  los símbolos por intersección de timestamps; no se rellenan huecos.
- **Rate limits.** Se respeta el limitador de ccxt y se reintenta con backoff, pero un
  exchange puede bloquear igual; el paper loop se corta tras N errores seguidos.
- **Paper en el reloj local.** Si el reloj del sistema está corrido, la detección de velas
  cerradas también.

### Licencia

Apache-2.0. Ver [LICENSE](LICENSE).

---

## EN

### Important notice

- **No profitable edge is claimed or guaranteed.** The three bundled strategies are
  standard teaching examples; in the real backtest documented below **all three lose
  money**, while buy & hold gained over the same period.
- A positive backtest would not prove anything either (overfitting, selection bias,
  regime change). This is not financial advice.
- Paper mode is a simulation. There is no real-order integration and adding one is out of
  scope for this repository.

### What it does

- **data**: ccxt public-data client (OHLCV, ticker) with exponential backoff + jitter,
  reconnection (rebuilds the ccxt instance after a transient error) and exchange fallback
  (default OKX → Binance → Kraken); historical paging; SQLite cache; polling feed that only
  exposes **closed** bars; deterministic synthetic data (GBM and regime-switching GBM).
- **strategies**: abstract `Strategy` + pluggable registry; `mean_reversion` (z-score /
  Bollinger-style), `momentum` (EMA crossover), `breakout` (Donchian channel + ATR margin).
- **backtest**: deterministic event-driven engine; the signal of bar *t* executes at the
  open of *t+1*; intrabar stops from high/low; total return, CAGR, annualized Sharpe and
  Sortino, max drawdown, win rate, profit factor, exposure, trade count, fees.
- **execution**: cost model (fee and slippage in bps), simulated portfolio, and a router
  shared by backtest and paper so both use the same logic.
- **risk**: fixed-fractional or volatility-target sizing, stop-loss / take-profit,
  per-symbol and total exposure caps, max-drawdown kill-switch, no re-entry after a stop
  until the signal changes.
- **paper**: live loop over REST polling, atomic JSON state, resume when the config
  fingerprint matches, graceful SIGINT/SIGTERM, bounded `--duration`, stop after N
  consecutive errors.
- **observability**: structured JSON logs and serializable counters/gauges.
- **config**: YAML files validated with pydantic (unknown keys are errors).

### What it does NOT do

No real orders or API keys; no order-book, market-impact, funding, borrow or liquidation
modelling; no websockets (REST polling only); no parameter optimization or walk-forward;
shorting is simulated but disabled by default.

### Run

```sh
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
market-lab fetch    --config configs/backtest_okx_1h.yaml
market-lab backtest --config configs/backtest_okx_1h.yaml
market-lab backtest --config configs/synthetic.yaml          # offline
market-lab paper    --config configs/paper_okx_1m.yaml --strategy momentum --duration 150
ruff check . && ruff format --check . && pytest                 # tests need no network
```

### Documented example

Real runs on 2026-10-30 (UTC); raw files in [`docs/examples/`](docs/examples/). The
backtest used public OKX 1h candles for BTC/USDT and ETH/USDT from 2026-05-01 00:00 to
2026-10-28 23:00 UTC (4344 bars each); see the table in the Spanish section. Results:
mean_reversion −2.46 %, momentum −9.44 %, breakout −14.87 % total return, against
buy & hold +5.40 % (equal weight, no costs). With 15 bps fees: −5.72 / −11.81 / −17.02 %;
with 35 bps: −17.74 / −20.67 / −24.13 %. The 150 s live paper run on OKX 1m data opened
simulated longs in BTC/USDT and ETH/USDT, closed BTC on a signal flip (PnL −16.51) and ended
at 9 976.66 equity with no errors; a second 45 s run resumed from the saved state, closed
ETH (PnL −12.94) and ended flat at 9 970.55 after 15.98 in total fees. A few minutes of paper
trading say nothing about strategy quality; they only show that the loop, risk controls,
persistence and resume work.

### Limitations

Simplified constant costs (default fee below OKX's published regular-user rate); no
exchange minimum order size or lot size enforcement; next-open execution; OHLC stop ambiguity resolved
conservatively (stop first); in the backtest a stop fills exactly at the stop level plus
slippage, and only a gap through the stop fills at the open (real stop orders can fill
worse); paper mode fills at the last traded price plus fixed slippage and does not model
the bid/ask spread; blocked entries are retried while the signal persists; no gap
filling; rate limits respected but not guaranteed; paper bar-close detection depends on the
local clock.

### License

Apache-2.0. See [LICENSE](LICENSE).
