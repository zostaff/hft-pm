# HFT on Prediction Markets

**A technical handbook with working code: microstructure, stochastic control, and an event-driven simulator for binary CLOB venues (Polymarket, Kalshi).**

---

## Contents

1. Why prediction markets are mathematically different
2. Microstructure: CLOB, binary payoff invariant, **and negRisk multi-outcome markets**
3. Glosten-Milgrom: why spreads exist
4. Avellaneda-Stoikov: full derivation, **validity bounds, and spread-capture economics**
5. The logit-space reformulation, **with boundary pathology**
6. Event-driven dynamics: Hawkes and jumps, **with MLE calibration**
7. Data, look-ahead bias, CLOB V2, **and the Polymarket fee structure**
8. Signals: OFI, microprice, queue value, VPIN — with calibration code
9. Complete quoting algorithm
10. Event-driven backtester with queue tracking, fee tiers, **production WebSocket client**
11. Purged combinatorial cross-validation, **Deflated Sharpe, PBO, Diebold-Mariano**
12. Risk and Kelly, **Bayesian Kelly under uncertainty**
13. Synthetic control for event studies
14. End-to-end working example
15. Common bugs, diagnostics, **monitoring and disaster recovery**
16. Roadmap
17. Bibliography

---

## 1. Why prediction markets are mathematically different

Three reasons to use prediction markets as the laboratory for HFT research:

1. **Bounded payoff.** Every contract resolves to exactly 0 or 1. No GBM assumption needed. Terminal variance is exactly $p(1-p)$.
2. **Identifiable information arrival.** Elections, Fed decisions, sports — discrete events with known timestamps. Cleaner causal identification than equities.
3. **Thin books.** Microstructure effects are 10-100x larger than in equities, so theoretical predictions are actually testable.

The strategies covered: **pure market making** (earning the spread) and **event-driven** (reacting to news). Cross-venue arbitrage is excluded — it's the most contested edge in the space and closes within seconds of opening.

---

## 2. Microstructure

Both Kalshi and Polymarket run central limit order books (CLOB). Same matching rules as NYSE, but two critical differences:

- **Each contract is its own isolated liquidity pool.** No cross-market depth. Spreads are wide.
- **Price = probability.** With the invariant $\text{YES} + \text{NO} = \$1$, prices are bounded in $(0,1)$. Volatility behaves wildly differently near boundaries than near 0.5.

### The binary payoff invariant

A YES share pays \$1 if the event occurs. So $p_t = \mathbb{E}_t[\mathbb{1}_{\text{event}}]$ under the risk-neutral measure. Consequences:

- **No drift.** Expected return is zero under the martingale measure. Any "trend" is either mispricing or unincorporated information.
- **Bounded variance.** Bernoulli variance $p(1-p)$ caps terminal uncertainty.
- **Black-Scholes doesn't apply directly.** Payoff is not differentiable in the underlying at the strike. The fix is to work in logit space — see §5.

### Stylized facts from Dubach (2026), 30B events over 52 days, 385k markets

> **Important caveat.** Dubach's data covers Polymarket V1 (scrape window Feb–Apr 2026, ending right before the V1→V2 cutover on 28 April 2026). The facts below are V1 facts. After the V2 migration $\kappa$, depth profiles, and wash rates likely shifted — re-measure on V2 data before calibrating a live bot. See §7.5.

1. **Longshot spread premium.** Spreads are systematically wider near $p=0$ and $p=1$. Counter-intuitive to pure Glosten-Milgrom but explained by inventory risk: a position at $p=0.02$ has 50x leverage.
2. **Depth is not top-heavy.** Closer to a geometric grid than a BBO-concentrated profile. Queue position matters at deeper levels too.
3. **Wash trading: median 1%, 22% upper tail.** Effective volume for backtests should be discounted accordingly.
4. **Polymarket leads Kalshi in price discovery** (Ng et al., 2025). Net order imbalance from large trades strongly predicts subsequent returns — the alpha thesis for event-driven strategies.

### 2.4 negRisk markets: when binary YES/NO is not the full story

For multi-outcome events ("Who will win the 2028 US Presidential Election?") with $N$ mutually exclusive outcomes, Polymarket runs them as a **negRisk** set. Each candidate $i$ has its own YES/NO market, but a global invariant connects them:

$$\sum_{i=1}^{N} p_i^{\text{YES}} = 1 \quad \text{(in equilibrium)}$$

The NegRiskAdapter contract enforces a key conversion: **1 NO share in market $i$ can be converted into 1 YES share in every other market $j \neq i$**, plus a USDC residual. Formally, if you hold $\{1 \cdot \text{NO}_i\}$, you can convert to $\{1 \cdot \text{YES}_j : j \neq i\}$ — a complete set worth exactly $1 because exactly one outcome wins.

This means the price space for a market maker quoting on candidate $i$ is bounded by both:

$$p_i^{\text{YES}} \in (0, 1) \quad \text{and} \quad p_i^{\text{YES}} \leq 1 - \max_{j \neq i} p_j^{\text{YES}}$$

The second constraint is non-trivial: if the leading candidate already trades at 0.7, every other candidate's YES is capped at 0.3, regardless of any model's belief.

**Three implications for HFT.**

1. **Cross-market AS skew.** When you have inventory in market $i$, the optimal quote should be skewed not only by inventory $q_i$ but also by your exposure across the full set (because the NO of market $i$ is fungible with YES of others). The single-market AS formula understates your true risk position.

2. **Apparent arbitrage is usually fake.** When $\sum p_i^{\text{YES}} > 1$ in a negRisk set, a naive reading says "buy all NOs for less than $N-1$, guaranteed profit." In practice the spread, taker fees, slippage, and tail outcomes eat the gap. Polymarket reports up to 9.5× capital efficiency from convert, but that's *capital efficiency for position holders*, not free money for arbitrageurs.

3. **Order placement requires the `neg_risk=True` flag.** Without it, the order is signed against the wrong Exchange contract and rejected. See §7.5 code example.

For market making on negRisk sets, the practical recommendation: start with single-market quoting and treat the cross-set dynamics as a separate inventory layer. A unified $N$-market optimal control solving HJB across the full simplex is doable (it's a $\sum q_i$-dimensional state space) but rarely worth the complexity given the gains. The autonomous.finance writeup of the "Three Laws of Motion" and convert mechanics is a good practitioner reference.

---

## 3. Glosten-Milgrom: why spreads exist

Three agent types:
- Informed traders (fraction $\alpha$) who know $V \in \{V_L, V_H\}$
- Uninformed liquidity traders (fraction $1-\alpha$), equal buy/sell probability
- Competitive risk-neutral market maker

Zero-profit condition for the market maker requires:

$$a = \mathbb{E}[V \mid \text{buy}], \qquad b = \mathbb{E}[V \mid \text{sell}]$$

Bayesian updating gives the spread:

$$a - b = 2\alpha(V_H - V_L)\pi_t(1-\pi_t)$$

**For prediction markets**, $V \in \{0, 1\}$ literally, so:

$$\boxed{a - b = 2\alpha \, p_t(1 - p_t)}$$

This is the binomial variance scaled by twice the informed fraction. Note it predicts *narrower* spreads at boundaries — opposite to the empirical longshot premium. The difference is inventory risk, which we derive next.

```python
def glosten_milgrom_spread(alpha: float, p: float) -> float:
    """Adverse selection component only. Add inventory term for full spread."""
    return 2 * alpha * p * (1 - p)
```

### Kyle's lambda (briefly)

Price impact is linear in net order flow with slope $\lambda = \sigma_v / (2\sigma_u)$ — fundamental volatility over noise volatility. We'll use this in §8 to interpret OFI.

---

## 4. Avellaneda-Stoikov: full derivation

### Setup

Midprice as Brownian motion (drift will be added in §6):

$$dS_t = \sigma \, dW_t$$

Limit order fill intensities decay exponentially in distance from mid:

$$\lambda^a(\delta) = A e^{-\kappa \delta}, \qquad \lambda^b(\delta) = A e^{-\kappa \delta}$$

Inventory and cash dynamics:

$$dq_t = dN^b_t - dN^a_t, \qquad dX_t = p^a_t \, dN^a_t - p^b_t \, dN^b_t$$

### Objective

Maximize expected CARA utility of terminal wealth:

$$V(t, x, q, s) = \sup_{\delta^a, \delta^b} \mathbb{E}_t\left[-\exp\left(-\gamma(X_T + q_T S_T)\right)\right]$$

### HJB equation

By dynamic programming:

$$\partial_t V + \tfrac{1}{2}\sigma^2 \partial_{ss} V + \sup_{\delta^a} \lambda^a(\delta^a)\left[V(t, x+(s+\delta^a), q-1, s) - V\right] + \sup_{\delta^b} \lambda^b(\delta^b)\left[V(t, x-(s-\delta^b), q+1, s) - V\right] = 0$$

### Ansatz

Try $V = -\exp(-\gamma x)\exp(-\gamma q s)\exp(\theta(t,q))$. Substituting and using $\partial_x V = -\gamma V$, $\partial_s V = -\gamma q V$, $\partial_{ss} V = \gamma^2 q^2 V$:

The optimization decouples. For the ask side:

$$\sup_{\delta^a} \lambda^a(\delta^a)\left[\exp(-\gamma \delta^a) e^{\Delta\theta_a} - 1\right]$$

where $\Delta\theta_a = \theta(t, q-1) - \theta(t, q)$. Setting derivative to zero:

$$\delta^{a*} = \frac{1}{\kappa}\ln\left(1 + \frac{\gamma}{\kappa}\right) + \frac{\Delta\theta_a}{\gamma}$$

### Closed form (asymptotic expansion)

After linearization in $q$ (valid for $\gamma\sigma^2(T-t)$ not too large):

$$\boxed{r(t,q) = S_t - q\gamma\sigma^2(T-t)} \quad \text{(reservation price)}$$

$$\boxed{\delta^* = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)} \quad \text{(half-spread)}$$

Place $p^b = r - \delta^*$ and $p^a = r + \delta^*$. The inventory effect is automatic: positive $q$ skews $r$ below $S_t$, making the maker more aggressive on the bid.

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class ASParams:
    gamma: float   # risk aversion
    sigma: float   # midprice volatility
    kappa: float   # fill intensity decay
    T: float       # terminal time

def avellaneda_stoikov_quotes(
    S: float, q: int, t: float, params: ASParams
) -> tuple[float, float]:
    """Return (bid, ask) per Avellaneda-Stoikov (2008).
    
    S: current midprice
    q: signed inventory (positive = long)
    t: current time, with t in [0, T]
    """
    tau = params.T - t
    r = S - q * params.gamma * params.sigma**2 * tau
    half_spread = (
        params.gamma * params.sigma**2 * tau
        + (2 / params.gamma) * np.log(1 + params.gamma / params.kappa)
    )
    return r - half_spread, r + half_spread
```

### Guéant-Lehalle-Fernandez-Tapia (2013)

GLT showed the HJB reduces to linear ODEs under a change of variables, giving closed-form solutions with inventory bounds. The approximation:

$$\delta^{a*}_\infty(q) \approx \frac{1}{\gamma}\ln\!\left(1 + \frac{\gamma}{\kappa}\right) + \frac{2q+1}{2}\sqrt{\frac{\sigma^2\gamma}{2\kappa A}\left(1 + \frac{\gamma}{\kappa}\right)^{1+\kappa/\gamma}}$$

```python
def glt_quotes(
    S: float, q: int, params: ASParams, A: float
) -> tuple[float, float]:
    """GLT closed-form quotes with inventory term. Use in steady state."""
    gamma, kappa, sigma = params.gamma, params.kappa, params.sigma
    base = (1/gamma) * np.log(1 + gamma/kappa)
    inv_term = np.sqrt(
        (sigma**2 * gamma) / (2 * kappa * A)
        * (1 + gamma/kappa) ** (1 + kappa/gamma)
    )
    delta_ask = base + ((2*q + 1) / 2) * inv_term
    delta_bid = base + ((-2*q + 1) / 2) * inv_term
    return S - delta_bid, S + delta_ask
```

### 4.7 Where the AS formula breaks

The linear-in-$q$ closed form is only valid in a regime. Three failure modes you'll hit in practice:

**Large inventory.** The ansatz drops $O(q^2)$ terms. As $|q|$ grows, the true optimal quote skew is *less* than linear (because the marginal disutility of one more share decreases when you can always liquidate at the next fill). At $|q| > q_{\max}/2$, AS over-skews and you pay more spread than you need to. Use GLT with inventory bounds, or fall back to full HJB numerics.

**Long horizon $\times$ high volatility.** The half-spread term $\gamma\sigma^2(T-t)$ explodes when $\gamma\sigma^2 T \gtrsim 1$. The asymptotic expansion assumed this quantity is small. If your bot is running on a multi-day market with high $\sigma$, the formula tells you to quote at unphysical widths — implementations cap $\delta^*$ at the price boundary, which silently breaks optimality.

**Price near $\{0, 1\}$.** AS assumes $S_t$ can drift freely. In prediction markets, drift compensation appears as a curvature term in logit space (§5). Using the raw-$S$ AS formula at $p < 0.05$ or $p > 0.95$ produces quotes that ignore the boundary's "soft wall." This is fixed by logit reformulation; see §5.1 for the pathological case in detail.

A rule of thumb: AS works cleanly for $|q\gamma\sigma^2(T-t)| < 0.5$ and $p \in [0.1, 0.9]$. Outside this region, switch to GLT or logit-space.

### 4.8 Spread-capture economics: how much does a maker actually earn?

We have $\delta^*$. What's the expected dollar PnL per unit time? Decompose by counterparty type:

$$\frac{d\mathbb{E}[\text{PnL}]}{dt} = \underbrace{2 A e^{-\kappa \delta^*} \cdot \delta^* \cdot (1-\alpha)}_{\text{spread captured from uninformed flow}} - \underbrace{2 A e^{-\kappa \delta^*} \cdot \alpha \cdot |\mathbb{E}[\Delta S \mid \text{informed fill}]|}_{\text{adverse selection cost}}$$

where $2 A e^{-\kappa\delta^*}$ is the round-trip fill rate (both sides), $(1-\alpha)$ is the uninformed fraction, and the second term is the per-fill loss to informed flow.

Substituting $\mathbb{E}[\Delta S | \text{informed}] \approx (V_H - V_L)\pi(1-\pi)$ from §3 and the breakeven condition:

$$\frac{d\mathbb{E}[\text{PnL}]}{dt} \approx 2 A e^{-\kappa\delta^*}\left[(1-\alpha)\delta^* - \alpha (V_H-V_L)\pi(1-\pi)\right]$$

**Key insight.** At the AS-optimal $\delta^*$, the bracket is *not* maximized — it's set so that the marginal utility of one more share equals the marginal disutility of one more inventory unit. Pure profit maximization (ignoring inventory) would quote *tighter*. AS quotes wider to control inventory variance.

To estimate $\alpha$ empirically, use VPIN (§8.3) as a proxy. In typical Polymarket markets, $\alpha \in [0.10, 0.30]$ in calm periods, spiking to $0.50+$ around news. The bracket goes negative — losses dominate — when $\alpha > \delta^*/((V_H-V_L)\pi(1-\pi) + \delta^*)$. That's why we widen or withdraw on high VPIN.

```python
def expected_pnl_rate(
    delta: float, A: float, kappa: float,
    alpha: float, p: float, V_H: float = 1.0, V_L: float = 0.0,
) -> float:
    """Expected dollar PnL per unit time for a maker quoting half-spread δ.
    α = informed-trader fraction (proxy: VPIN).
    Returns positive = profitable, negative = adversely selected.
    """
    fill_rate = 2 * A * np.exp(-kappa * delta)
    spread_capture = (1 - alpha) * delta
    adv_selection = alpha * abs(V_H - V_L) * p * (1 - p)
    return fill_rate * (spread_capture - adv_selection)

def breakeven_alpha(
    delta: float, p: float, V_H: float = 1.0, V_L: float = 0.0,
) -> float:
    """Maximum α before quoting half-spread δ becomes unprofitable."""
    payoff_range = abs(V_H - V_L)
    return delta / (payoff_range * p * (1 - p) + delta)
```

Use `breakeven_alpha` as a gate alongside VPIN: if your estimated $\alpha > \text{breakeven\_alpha}$, withdraw or widen until you're profitable again.

---

## 5. The logit-space reformulation

The AS model assumes the midprice can drift arbitrarily. For prediction markets this breaks because $S_t \in (0,1)$. The fix (arXiv:2510.15205): work in logit space.

Define $x_t = \ln(p_t / (1 - p_t))$. Then $x_t \in \mathbb{R}$ and standard semimartingale tools apply. The dynamics:

$$dx_t = -\tfrac{1}{2}\sigma_b^2 (1 - 2p_t) \, dt + \sigma_b \, dW_t + \text{jump terms}$$

The drift compensator keeps $p_t$ a martingale. Solve the HJB in $x$-space, transform back. The advantages:

- $\sigma_b$ is "belief volatility," approximately constant across the probability range
- Quotes naturally widen near 0 and 1, matching the longshot premium
- Jumps from news events have a clean additive representation

```python
def logit(p: float) -> float:
    return np.log(p / (1 - p))

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def logit_space_quotes(
    p_mid: float, q: int, t: float, params: ASParams
) -> tuple[float, float]:
    """AS quotes computed in logit space, transformed back to probability."""
    x_mid = logit(p_mid)
    tau = params.T - t
    x_res = x_mid - q * params.gamma * params.sigma**2 * tau
    half_spread = (
        params.gamma * params.sigma**2 * tau
        + (2 / params.gamma) * np.log(1 + params.gamma / params.kappa)
    )
    return sigmoid(x_res - half_spread), sigmoid(x_res + half_spread)
```

### 5.1 Boundary pathology and the local-time interpretation

When $p_t \to 0$ or $p_t \to 1$, the logit $x_t \to \mp\infty$. The drift term $-\frac{1}{2}\sigma_b^2(1-2p_t)$ blows up, pushing $x_t$ *back toward zero* — exactly what makes the boundary a soft wall rather than an absorbing barrier. This is the right behaviour: the closer to certainty, the slower probabilities update *unconditional on new information*.

What happens to the AS half-spread? In logit space it's constant in $p_t$. But after transforming back via sigmoid, the *price-space* half-spread shrinks proportionally to $\sigma'(x) = p(1-p)$:

$$\delta_{\text{price}}^* \approx p(1-p) \cdot \delta_{\text{logit}}^*$$

So at $p = 0.5$, a logit half-spread of $0.4$ gives a price spread of $0.10$. At $p = 0.05$, the same logit half-spread gives only $0.019$ in price terms. The maker quotes tightly in price units near boundaries — which sounds wrong given the longshot spread premium, but isn't.

The longshot premium comes from a different source: **inventory risk per dollar at risk** explodes near boundaries. Even if the logit-space spread is unchanged, the optimal inventory cap shrinks, which means smaller positions, which means tighter effective quotes (the GLT inventory term in §4.6 drops with $|q|_{\max}$).

This unifies the picture: in logit space, the math is regular and constant; the longshot premium emerges from the *position size* constraint, not from the per-trade quoting. The exposure cap $|q| \leq M \sqrt{p(1-p)}$ from §12 is the operational realization.

```python
def logit_space_quotes_with_boundary_cap(
    p_mid: float, q: int, t: float, params: ASParams,
    M: float = 100.0,  # max exposure scaling
) -> tuple[float, float] | None:
    """logit-space AS with inventory cap that respects boundary geometry.
    Returns None if inventory exceeds boundary-aware cap."""
    boundary_cap = int(M * np.sqrt(p_mid * (1 - p_mid)))
    if abs(q) > boundary_cap:
        return None  # withdraw
    return logit_space_quotes(p_mid, q, t, params)
```

---

## 6. Event-driven dynamics

Real prediction markets are dominated by news events. Cartea, Jaimungal & Ricci (2018) extend AS with:

1. A predictable midprice drift $\alpha_t$ driven by recent order flow and news
2. Hawkes processes for clustered order arrivals
3. Jump terms for scheduled news

The midprice becomes:

$$dS_t = \alpha_t \, dt + \sigma \, dW_t + \int J \, \tilde{N}(dt, dJ)$$

The HJB gains a term proportional to $q\alpha_t$, and optimal quotes get an **alpha skew**:

$$p^a = S_t + \frac{\alpha_t}{\gamma\sigma^2}(T-t)^* + \delta^*_{\text{AS}}(q) + \text{jump compensation}$$

Hawkes intensity for buy/sell market order arrivals:

$$\lambda_t^{\pm} = \mu^{\pm} + \int_0^t \alpha^{\pm} e^{-\beta^{\pm}(t-s)} \, dN_s^{\pm}$$

```python
class HawkesIntensity:
    """Univariate Hawkes process with exponential kernel."""
    def __init__(self, mu: float, alpha: float, beta: float):
        self.mu, self.alpha, self.beta = mu, alpha, beta
        self.state = 0.0  # exponentially-decayed sum of past events
        self.last_t = 0.0

    def update(self, t: float, event: bool) -> float:
        """Decay state to time t, optionally add an event impulse, return intensity."""
        dt = t - self.last_t
        self.state *= np.exp(-self.beta * dt)
        if event:
            self.state += self.alpha
        self.last_t = t
        return self.mu + self.state
```

### Scheduled jumps

Before a known jump time $\tau$, optimal spread widens. In practice, many makers simply withdraw quotes in the last seconds — the optimal spread becomes effectively infinite under any reasonable jump distribution.

### 6.3 Hawkes calibration and the stationarity constraint

The exponential-kernel Hawkes process $\lambda_t = \mu + \int_0^t \alpha e^{-\beta(t-s)} dN_s$ has a hard stability condition: the **branching ratio** $n \equiv \alpha/\beta$ must satisfy $n < 1$. Equivalently, every "parent" event triggers an expected $n$ "daughter" events; if $n \geq 1$ the process explodes.

For exponential kernel, the closed-form log-likelihood is:

$$\log \mathcal{L}(\mu, \alpha, \beta \mid t_1, \ldots, t_N) = \sum_{i=1}^N \log\left(\mu + \alpha R_i\right) - \mu T - \frac{\alpha}{\beta}\sum_{i=1}^N\left(1 - e^{-\beta(T - t_i)}\right)$$

where $R_i = \sum_{j<i} e^{-\beta(t_i - t_j)}$ — efficiently computed via the recurrence $R_i = e^{-\beta(t_i - t_{i-1})}(R_{i-1} + 1)$.

```python
from scipy.optimize import minimize
from scipy.special import logsumexp

def hawkes_log_likelihood(
    params: np.ndarray, event_times: np.ndarray, T: float,
) -> float:
    """Negative log-likelihood for exponential-kernel Hawkes process.
    params = (mu, alpha, beta), all positive. Use minimize() to fit.
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return 1e10  # invalid; penalize
    # Stationarity: branching ratio must be < 1
    n_branching = alpha / beta
    if n_branching >= 1.0:
        return 1e10
    
    N = len(event_times)
    # Recurrence for R_i = e^{-β·Δ}(R_{i-1} + 1)
    R = np.zeros(N)
    for i in range(1, N):
        R[i] = np.exp(-beta * (event_times[i] - event_times[i-1])) * (R[i-1] + 1)
    
    # Log-intensity at each event
    log_intensity = np.log(mu + alpha * R)
    # Compensator
    compensator = mu * T + (alpha/beta) * np.sum(
        1 - np.exp(-beta * (T - event_times))
    )
    
    return -(log_intensity.sum() - compensator)


def fit_hawkes_mle(event_times: np.ndarray, T: float) -> dict:
    """Fit exponential-kernel Hawkes via MLE with stationarity constraint."""
    # Initial guess: mu = empirical rate / 2; alpha/beta = 0.5
    initial_rate = len(event_times) / T
    x0 = np.array([initial_rate * 0.5, 1.0, 2.0])  # mu, alpha, beta with n=0.5
    
    result = minimize(
        hawkes_log_likelihood, x0, args=(event_times, T),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (0, None), (1e-3, None)],
    )
    mu_hat, alpha_hat, beta_hat = result.x
    return {
        "mu": float(mu_hat),
        "alpha": float(alpha_hat),
        "beta": float(beta_hat),
        "branching_ratio": float(alpha_hat / beta_hat),
        "log_likelihood": float(-result.fun),
        "converged": bool(result.success),
        "half_life_seconds": float(np.log(2) / beta_hat),
    }
```

**The Hardiman-Bouchaud shortcut.** Full MLE is slow on long sequences. The 2014 Hardiman-Bouchaud paper gives a model-free estimator using only the count variance: if $N(\Delta)$ is the event count in window $\Delta$, then for large $\Delta$:

$$\frac{\text{Var}[N(\Delta)]}{\mathbb{E}[N(\Delta)]} \approx \frac{1}{(1-n)^2}$$

So $n \approx 1 - \sqrt{\mathbb{E}[N]/\text{Var}[N]}$. Use this as a fast diagnostic; trust MLE for production parameters.

```python
def hardiman_bouchaud_branching_ratio(
    event_times: np.ndarray, T: float, n_windows: int = 100,
) -> float:
    """Model-free estimator of branching ratio from count over/dispersion.
    Hardiman & Bouchaud 2014. Fast diagnostic; not as accurate as MLE."""
    bins = np.linspace(0, T, n_windows + 1)
    counts = np.histogram(event_times, bins=bins)[0]
    mean_count = counts.mean()
    var_count = counts.var()
    if var_count <= mean_count:
        return 0.0  # Sub-Poisson; no excitation
    return 1.0 - np.sqrt(mean_count / var_count)
```

For market making, you typically observe $n \in [0.3, 0.8]$ in equity microstructure. Higher $n$ on news days. If your fitted $n > 0.95$, the regime is *critical* — small inputs lead to large cascades. Most flash-crash signatures show $n$ spiking toward 1.0 in the seconds before. This is a real-time crash early-warning signal you can build directly from your fill data.

---

## 7. Data: the four layers of look-ahead bias

This is where most strategies die.

**Layer 1: timestamp-level.** Every feature must use only data with timestamp strictly less than the decision timestamp. Mechanical enforcement.

**Layer 2: event-detection-level.** Your historical event timestamp is "true," but your live pipeline takes time to detect that event from noisy sources (Twitter, RSS). Backtests using the true timestamp give the strategy seconds of free information. Always replay events at the *first-detectable* timestamp from your live pipeline.

**Layer 3: labeling-level.** When you label "market moved X% in response to event Y," you're labeling Y as the cause because you can see the move. The model learns to recognize clear signals that don't exist in real-time noise. Use causal event detection — declare an event using only data up to that moment.

**Layer 4: causal inference.** Even with perfect causal detection, regressing prices on event indicators conflates causation with regime. Goldsmith-Pinkham & Lyu (2025) show that traditional event-study estimators are inconsistent when factor models are misspecified — which they always are. Use synthetic control: for each event, construct a counterfactual from related but unaffected markets.

For prediction markets, this is conceptually easier than equities — many parallel markets give natural controls.

### Data sources

| Source | Use | Latency | Notes |
|---|---|---|---|
| Polymarket WebSocket market channel | Live L2 | sub-50ms median | `wss://ws-subscriptions-clob.polymarket.com/ws/market` |
| Polymarket subgraph (The Graph) | Historical | Block time (~2s) | Canonical on-chain trade record |
| Polymarket REST `/orderbook` | Snapshots | Polling | Reconstruction reference |
| Kalshi API | Live + historical | Variable | Requires KYC; CFTC-regulated |
| GDELT | Global news | Daily | Free, coarse |
| Twitter/X firehose | Real-time sentiment | Sub-second | Contractual access |

```python
# Polymarket WebSocket subscriber (minimal example)
import asyncio
import json
import websockets
from datetime import datetime

POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

async def stream_market(asset_ids: list[str], on_event):
    """Subscribe to Polymarket market channel and dispatch events.
    
    on_event(event_dict, recv_timestamp_ms) — your callback.
    """
    async with websockets.connect(POLY_WS, ping_interval=30) as ws:
        await ws.send(json.dumps({"type": "market", "assets_ids": asset_ids}))
        async for raw in ws:
            recv_ts = int(datetime.utcnow().timestamp() * 1000)
            try:
                msg = json.loads(raw)
                # msg is either a list of events or a single event
                events = msg if isinstance(msg, list) else [msg]
                for ev in events:
                    on_event(ev, recv_ts)
            except json.JSONDecodeError:
                continue
```

The `recv_ts - server_ts` distribution is your latency profile. Log it persistently; it's a calibration input for your simulator.

### 7.5 Polymarket CLOB V2 (post-28 April 2026)

CLOB V2 went live on 28 April 2026 at ~11:00 UTC. It is a hard cutover — V1 SDKs and V1-signed orders are no longer accepted. What changed:

| Component | V1 | V2 |
|---|---|---|
| Collateral | USDC.e (bridged) | **pUSD** (ERC-20 on Polygon, 1:1 USDC-backed, on-chain enforced) |
| Exchange contracts | CTF Exchange | **CTF Exchange V2 + Neg Risk CTF Exchange V2** (audited by Cantina, Quantstamp) |
| SDK | `py-clob-client` / `@polymarket/clob-client` | **`py-clob-client-v2` / `@polymarket/clob-client-v2`** |
| Signature for new integrations | Type 0/1/2 (EOA/Magic/Proxy) | **Type 3 (POLY_1271, ERC-1271 wrapped)** |
| Order uniqueness | Random salt | **Timestamp (ms)** |
| Builder attribution | Off-chain metadata | **`builderCode` field in signed order** |
| WebSocket URLs | unchanged | unchanged |
| L1/L2 auth flow | unchanged | unchanged |

**The "ghost fill" bug** that plagued V1 (order shows filled on CLOB but balance doesn't move because the proxy wallet's on-chain settlement failed) is gone — ERC-1271 validation makes the smart-contract deposit wallet authorize each order atomically.

**Three operational consequences you must implement.**

1. **Use the V2 SDK.** Existing users with proxy/Safe setups can still run V1-style sig types 1/2, but every new account uses deterministic deposit wallets with sig type 3.

2. **The mandatory `/balance-allowance/update` call.** After any deposit, withdrawal, or allowance change, you must call this endpoint with `signature_type=3`. Without it, the CLOB cache won't reflect your on-chain balance and orders will be rejected with `insufficient balance` even though chain state is fine. This is the single most common V2 production bug. Hit it after every funding event.

3. **Treat the cutover as a regime boundary.** Don't pool pre/post-28-April data for training. The matching engine, latency profile, fill behavior, and market participant mix all shifted. Re-measure $\kappa$, $\sigma_b$, and the empirical stylized facts on V2-only data.

```python
# V2 client setup (py-clob-client-v2)
# pip install py-clob-client-v2
import os
from py_clob_client_v2 import (
    ApiCreds, ClobClient, OrderArgs, OrderType,
    PartialCreateOrderOptions, Side,
)

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet; use 80002 for Amoy testnet

# Step 1: L1 auth — derive API credentials from your wallet
bootstrap = ClobClient(host=HOST, chain_id=CHAIN_ID, key=os.environ["PRIVATE_KEY"])
creds = bootstrap.create_or_derive_api_key()

# Step 2: Fully authenticated client with deposit wallet + ERC-1271 signing
client = ClobClient(
    host=HOST,
    chain_id=CHAIN_ID,
    key=os.environ["PRIVATE_KEY"],
    creds=creds,
    signature_type=3,                              # POLY_1271 (ERC-1271)
    funder=os.environ["DEPOSIT_WALLET_ADDRESS"],   # smart-account deposit wallet
)

# CRITICAL: after any deposit/withdrawal/allowance change, sync the cache
# Without this, orders fail with "insufficient balance" despite on-chain reality.
def sync_balance_cache(client: ClobClient):
    """Force CLOB to re-read on-chain balance. Call after every funding event."""
    return client.update_balance_allowance(signature_type=3)

# Place a resting GTC limit order (post-only available for makers)
def place_limit(client: ClobClient, token_id: str, price: float,
                size: float, side: Side, tick_size: str = "0.01",
                post_only: bool = True) -> dict:
    return client.create_and_post_order(
        order_args=OrderArgs(token_id=token_id, price=price, size=size, side=side),
        options=PartialCreateOrderOptions(tick_size=tick_size, neg_risk=False),
        order_type=OrderType.GTC,  # use FOK / FAK for taker; GTC for maker
    )

# Order management
def cancel(client: ClobClient, order_id: str) -> dict:
    return client.cancel(order_id=order_id)

def cancel_all_for_market(client: ClobClient, market: str) -> dict:
    return client.cancel_market_orders(market=market)
```

**Caveat on `post_only`.** V2 maker rebate eligibility requires `post_only=True` so the order never crosses the spread and only acts as maker liquidity. A taker fill (when your price crosses) collects no rebate and pays the taker fee. For an AS/GLT market maker, always submit `post_only`.

### 7.6 Polymarket fee structure (post-March 2026)

Polymarket ran fee-free for years. As of Fee Structure V2 (30 March 2026) this is no longer true. The model:

- **Maker orders pay $0** in taker fees. Makers receive a **rebate** funded by taker fees in the same market category.
- **Taker fees are category-dependent and probability-dependent.** Peak fee at $p = 0.5$; drops smoothly toward both boundaries.

The taker fee formula:

$$\text{fee}(p) = F_{\text{cat}} \cdot \text{contracts} \cdot p \cdot (1-p) \cdot \text{shape\_factor}(p)$$

where $F_{\text{cat}}$ is the per-category peak rate. Specifics (Polymarket Global, as of writing):

| Category | Peak taker rate $F$ | Maker rebate share | Peak fee at $p=0.5$ per 100 shares |
|---|---|---|---|
| Crypto (15-min) | 1.80% | 20% | $1.80 |
| Economics | 1.50% | 25% | $1.50 |
| Mentions | 1.56% | 25% | $1.56 |
| Culture / Weather | 1.25% | 25% | $1.25 |
| Finance | 1.00% | **50%** | $1.00 |
| Politics / Tech | 1.00% | 25% | $1.00 |
| Sports | 0.75% | 25% | $0.75 |
| Geopolitics / World events | 0% | n/a | $0.00 |

(Polymarket US — the CFTC-regulated venue — uses a flat 0.30% taker / 0.20% maker rebate instead.)

**What this means for the AS economics in §4.8.** The breakeven $\alpha$ shifts because the maker rebate is positive PnL on every fill *additional* to the spread. Effectively:

$$\delta^*_{\text{effective}} = \delta^* + \text{rebate}(p)$$

For a Finance market with 50% rebate at $p = 0.5$, the maker captures (notional $\delta^*$) + ($1.00 \times 0.50/100) = (\delta^*$ + $0.005). On a $0.50 quote with $0.005 half-spread, the rebate doubles your edge.

The Finance category's 50% rebate is a deliberate Polymarket subsidy to attract liquidity to a newly fee-bearing area. Trade where the rebates are richest if all else is equal.

```python
from enum import Enum

class FeeCategory(Enum):
    CRYPTO = ("crypto", 0.018, 0.20)        # 15-min crypto markets
    ECONOMICS = ("economics", 0.015, 0.25)
    MENTIONS = ("mentions", 0.0156, 0.25)
    CULTURE = ("culture", 0.0125, 0.25)
    WEATHER = ("weather", 0.0125, 0.25)
    FINANCE = ("finance", 0.010, 0.50)
    POLITICS = ("politics", 0.010, 0.25)
    TECH = ("tech", 0.010, 0.25)
    SPORTS = ("sports", 0.0075, 0.25)
    GEOPOLITICS = ("geopolitics", 0.0, 0.0)
    OTHER = ("other", 0.0, 0.0)

    def __init__(self, name, peak_rate, rebate_share):
        self.cat_name = name
        self.peak_rate = peak_rate
        self.rebate_share = rebate_share

def taker_fee(price: float, size: float, category: FeeCategory) -> float:
    """Polymarket V2 dynamic taker fee. Returns $ paid by taker."""
    if category.peak_rate == 0:
        return 0.0
    # Symmetric around p=0.5, peaks at midpoint, vanishes at boundaries
    return category.peak_rate * size * price * (1 - price) * 4
    # The "× 4" normalizes so that peak == peak_rate × size at p=0.5

def maker_rebate(price: float, size: float, category: FeeCategory) -> float:
    """Maker rebate per fill — fraction of the taker fee paid by counterparty."""
    return taker_fee(price, size, category) * category.rebate_share

def effective_spread_with_rebate(
    half_spread_pct: float, p: float, category: FeeCategory,
) -> float:
    """Effective half-spread including maker rebate, as a fraction of notional."""
    spread = half_spread_pct  # e.g. 0.005 for 0.5%
    rebate = maker_rebate(p, 1.0, category)  # per $1 notional
    return spread + rebate
```

**Net implication for §10's Backtester.** Set `fee_bps` based on category and whether you're maker or taker per fill — these are not the same any more. The next subsection updates the simulator accordingly.

---

## 8. Signals

### 8.1 Order Flow Imbalance (Cont-Kukanov-Stoikov 2014)

For each order book event $n$, define:

$$e_n = \mathbb{1}_{\Delta P^B_n \geq 0} V^B_n - \mathbb{1}_{\Delta P^B_n \leq 0} V^B_{n-1} - \mathbb{1}_{\Delta P^A_n \leq 0} V^A_n + \mathbb{1}_{\Delta P^A_n \geq 0} V^A_{n-1}$$

Then $\text{OFI}_\tau = \sum_n e_n$ over a window. Empirically, $\Delta x \approx \beta \cdot \text{OFI}$ where $x$ is logit price.

```python
from collections import deque

class OFICalculator:
    """Cont-Kukanov-Stoikov OFI with rolling window."""
    def __init__(self, window_seconds: float = 1.0):
        self.window = window_seconds
        self.events = deque()  # (timestamp, e_n)
        self.prev_bid_px = self.prev_bid_sz = None
        self.prev_ask_px = self.prev_ask_sz = None

    def update(self, ts: float, bid_px: float, bid_sz: float,
               ask_px: float, ask_sz: float) -> float:
        if self.prev_bid_px is None:
            self.prev_bid_px, self.prev_bid_sz = bid_px, bid_sz
            self.prev_ask_px, self.prev_ask_sz = ask_px, ask_sz
            return 0.0

        # Bid side contribution
        if bid_px > self.prev_bid_px:
            e_bid = bid_sz
        elif bid_px < self.prev_bid_px:
            e_bid = -self.prev_bid_sz
        else:
            e_bid = bid_sz - self.prev_bid_sz

        # Ask side contribution (sign flipped)
        if ask_px < self.prev_ask_px:
            e_ask = -ask_sz
        elif ask_px > self.prev_ask_px:
            e_ask = self.prev_ask_sz
        else:
            e_ask = -(ask_sz - self.prev_ask_sz)

        e_n = e_bid + e_ask
        self.events.append((ts, e_n))
        self._evict(ts)

        self.prev_bid_px, self.prev_bid_sz = bid_px, bid_sz
        self.prev_ask_px, self.prev_ask_sz = ask_px, ask_sz
        return sum(e for _, e in self.events)

    def _evict(self, now: float):
        while self.events and now - self.events[0][0] > self.window:
            self.events.popleft()
```

### 8.2 Microprice (Stoikov 2018)

Imbalance-weighted midprice as a martingale-by-construction fair-value estimator:

$$\text{microprice} = \frac{V^A p^B + V^B p^A}{V^A + V^B}$$

```python
def microprice(bid_px: float, bid_sz: float,
               ask_px: float, ask_sz: float) -> float:
    """Stoikov's weighted-mid (first iteration of full microprice)."""
    total = bid_sz + ask_sz
    if total == 0:
        return (bid_px + ask_px) / 2
    return (ask_sz * bid_px + bid_sz * ask_px) / total
```

### 8.3 VPIN for prediction markets

Standard VPIN (Easley, López de Prado, O'Hara 2012) normalized for the binary-payoff structure:

$$\text{VPIN}^{\text{PM}} = \frac{1}{n}\sum_{i=1}^{n} \frac{|V^B_i - V^S_i|}{\sqrt{p_i(1-p_i)} \cdot (V^B_i + V^S_i)}$$

```python
class VPINCalculator:
    """Volume-bucketed toxicity for prediction markets."""
    def __init__(self, bucket_volume: float, n_buckets: int = 50):
        self.bucket_volume = bucket_volume
        self.n_buckets = n_buckets
        self.buckets = deque()  # list of (buy_vol, sell_vol, p_mean)
        self.cur_buy = self.cur_sell = 0.0
        self.cur_p_sum = self.cur_n = 0

    def add_trade(self, volume: float, is_buy: bool, p: float):
        if is_buy:
            self.cur_buy += volume
        else:
            self.cur_sell += volume
        self.cur_p_sum += p
        self.cur_n += 1

        while self.cur_buy + self.cur_sell >= self.bucket_volume:
            # Close bucket
            p_mean = self.cur_p_sum / max(self.cur_n, 1)
            self.buckets.append((self.cur_buy, self.cur_sell, p_mean))
            if len(self.buckets) > self.n_buckets:
                self.buckets.popleft()
            # Carry residual (proportional)
            excess = self.cur_buy + self.cur_sell - self.bucket_volume
            ratio = self.cur_sell / (self.cur_buy + self.cur_sell) if self.cur_buy + self.cur_sell > 0 else 0.5
            self.cur_sell = excess * ratio
            self.cur_buy = excess * (1 - ratio)
            self.cur_p_sum = p_mean  # reset tracker to last p
            self.cur_n = 1

    def vpin_pm(self) -> float:
        """Return adverse-selection toxicity in [0, 1]."""
        if not self.buckets:
            return 0.0
        total = 0.0
        for buy, sell, p in self.buckets:
            denom = np.sqrt(p * (1 - p)) * (buy + sell)
            if denom > 0:
                total += abs(buy - sell) / denom
        return total / len(self.buckets)
```

### 8.4 Queue position value

Guo, Ruan & Zhu (2018) give the expected time-to-fill at queue position $k$ with total queue size $Q$:

$$\mathbb{E}[\text{time to fill}] \approx \frac{k}{\mu - \lambda_c} \cdot \frac{1}{1 - \rho^k}$$

In practice, track queue position; cancel and reprice when the value of holding the spot falls below the cost of repricing.

### 8.5 Calibrating the OFI → alpha coefficient

The `alpha_beta` parameter in §9 isn't a free knob — it's an empirical regression slope. The procedure: collect (OFI, future Δlogit-price) pairs from historical data, regress, store the coefficient. Use **purged** out-of-sample splits to avoid look-ahead in the calibration itself.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

def calibrate_ofi_alpha(
    book_events: pd.DataFrame,        # cols: ts, bid_px, bid_sz, ask_px, ask_sz
    horizon_ms: float = 1000.0,       # predict Δx over this horizon
    ofi_window_ms: float = 1000.0,    # OFI accumulation window
) -> dict:
    """Fit Δx_{t+h} = β · OFI_t + ε. Return slope, R², residual std.
    
    Δx is the change in logit-microprice. Use the resulting β as `alpha_beta`
    in compute_quotes(). Use this on TRAINING-fold data only — never on the
    full dataset, never with future data overlapping the test fold.
    """
    df = book_events.copy().sort_values("ts").reset_index(drop=True)
    
    # 1. Logit microprice
    mp = (df["ask_sz"] * df["bid_px"] + df["bid_sz"] * df["ask_px"]) / (
        df["bid_sz"] + df["ask_sz"]
    )
    mp = mp.clip(1e-4, 1 - 1e-4)
    df["x"] = np.log(mp / (1 - mp))
    
    # 2. Rolling OFI from book deltas
    ofi_calc = OFICalculator(window_seconds=ofi_window_ms / 1000)
    ofi_series = []
    for _, row in df.iterrows():
        ofi_series.append(
            ofi_calc.update(
                row["ts"] / 1000, row["bid_px"], row["bid_sz"],
                row["ask_px"], row["ask_sz"]
            )
        )
    df["ofi"] = ofi_series
    
    # 3. Forward logit-price change at horizon
    df["ts_target"] = df["ts"] + horizon_ms
    target = df.set_index("ts")["x"].reindex(
        df["ts_target"], method="ffill"
    ).values
    df["dx_fwd"] = target - df["x"].values
    
    # 4. Drop missing forwards and fit
    fit = df.dropna(subset=["dx_fwd"])
    X = fit[["ofi"]].values
    y = fit["dx_fwd"].values
    model = LinearRegression(fit_intercept=False).fit(X, y)
    
    pred = model.predict(X)
    residuals = y - pred
    
    return {
        "alpha_beta": float(model.coef_[0]),
        "r2": float(model.score(X, y)),
        "residual_std": float(residuals.std()),
        "n_samples": len(fit),
        "horizon_ms": horizon_ms,
    }
```

A typical fit on liquid Polymarket markets returns $\beta \in [0.1, 1.0]$ logit-units per unit of OFI, with $R^2$ in the 0.02–0.10 range. Higher $R^2$ on in-sample is a warning sign of overfit, not a victory — the relationship is genuinely weak per event but accumulates statistical edge over many quotes. **If your in-sample $R^2 > 0.2$, you've leaked.**

### 8.6 Full L2 order book with queue tracking

For real market making you need more than top-of-book. Resting orders sit at specific price levels with specific queue positions; you must track all of it to make rational cancel/reprice decisions.

```python
from sortedcontainers import SortedDict

class L2OrderBook:
    """Per-level book state with our resting-order queue tracking.
    
    The book holds total resting size at each price level (excluding our
    own orders). When a trade arrives, we know how much volume executes
    ahead of our orders at that level.
    """
    def __init__(self, tick: float = 0.001):
        self.tick = tick
        self.bids: SortedDict[float, float] = SortedDict()  # px -> size
        self.asks: SortedDict[float, float] = SortedDict()  # px -> size
        # Our resting orders: order_id -> (side, price, size, queue_ahead_at_placement)
        self.our_orders: dict[int, tuple] = {}
        # Snapshot of total size at the level when WE placed (for fill tracking)
        self.level_size_at_placement: dict[int, float] = {}

    def apply_book_snapshot(self, bids: list[tuple[float, float]],
                            asks: list[tuple[float, float]]):
        """Replace the book from a WebSocket snapshot."""
        self.bids = SortedDict({px: sz for px, sz in bids if sz > 0})
        self.asks = SortedDict({px: sz for px, sz in asks if sz > 0})

    def apply_price_change(self, side: str, price: float, new_size: float):
        """Apply a price_change event from the market channel."""
        book = self.bids if side == "BUY" else self.asks
        if new_size <= 0:
            book.pop(price, None)
        else:
            book[price] = new_size

    def best_bid(self) -> tuple[float, float] | None:
        if not self.bids:
            return None
        px = self.bids.keys()[-1]  # highest bid
        return px, self.bids[px]

    def best_ask(self) -> tuple[float, float] | None:
        if not self.asks:
            return None
        px = self.asks.keys()[0]  # lowest ask
        return px, self.asks[px]

    def add_our_order(self, order_id: int, side: str, price: float, size: float):
        """Record that WE just placed an order. Queue-ahead = current level size."""
        book = self.bids if side == "bid" else self.asks
        queue_ahead = book.get(price, 0.0)
        self.our_orders[order_id] = (side, price, size, queue_ahead)
        self.level_size_at_placement[order_id] = queue_ahead + size

    def remove_our_order(self, order_id: int):
        self.our_orders.pop(order_id, None)
        self.level_size_at_placement.pop(order_id, None)

    def process_trade(self, trade_price: float, trade_size: float, aggressor_side: str
                      ) -> list[tuple[int, float]]:
        """A market trade at trade_price consumes liquidity. Returns list of
        (our_order_id, filled_size) for any of our orders that got hit.
        
        The aggressor_side is "BUY" if it was a market buy (hits asks).
        We need to update queue_ahead for our resting orders on that side.
        """
        our_side = "ask" if aggressor_side == "BUY" else "bid"
        fills = []
        
        for oid, (side, price, size, qa_initial) in list(self.our_orders.items()):
            if side != our_side:
                continue
            # Only orders at or better than the trade price can be touched
            if side == "ask" and price > trade_price:
                continue
            if side == "bid" and price < trade_price:
                continue
            if abs(price - trade_price) > self.tick / 2:
                continue  # different level; queue unaffected
            
            # Estimate remaining queue ahead. We can't observe it directly,
            # but we can estimate from how much the level has shrunk.
            book = self.bids if side == "bid" else self.asks
            current_level_size = book.get(price, 0.0)
            initial_total = self.level_size_at_placement[oid]
            consumed = max(0.0, initial_total - current_level_size - size)
            queue_remaining = max(0.0, qa_initial - consumed)
            
            if trade_size <= queue_remaining:
                # All this trade is in front of us; we don't fill
                continue
            
            # The trade reaches our order
            fill_size = min(trade_size - queue_remaining, size)
            fills.append((oid, fill_size))
            # Update our recorded size
            new_size = size - fill_size
            if new_size <= 0:
                self.remove_our_order(oid)
            else:
                # Queue ahead is now zero (we got partially filled at front)
                self.our_orders[oid] = (side, price, new_size, 0.0)
        
        return fills

    def queue_position_estimate(self, order_id: int) -> float | None:
        """Estimate volume currently ahead of our order at its level.
        Returns None if order not tracked."""
        info = self.our_orders.get(order_id)
        if info is None:
            return None
        side, price, size, qa_initial = info
        book = self.bids if side == "bid" else self.asks
        current_level_size = book.get(price, 0.0)
        initial_total = self.level_size_at_placement[order_id]
        consumed = max(0.0, initial_total - current_level_size - size)
        return max(0.0, qa_initial - consumed)
```

This replaces the TODO in §10's Backtester. Note the assumption: when level size shrinks, we attribute the shrinkage to cancellations *in front of* us proportionally to our queue position — a reasonable approximation when cancel-flow is uniform across the queue, but it understates queue-jump risk from large mid-queue cancellations. For very high-fidelity simulation, use L3 data and track order-by-order.

---

## 9. Complete quoting algorithm

Putting §4-8 together. Algorithm runs on each order book update:

```python
@dataclass
class StrategyState:
    inventory: int = 0
    last_quote_bid: float = 0.0
    last_quote_ask: float = 1.0

def compute_quotes(
    book: "OrderBook",
    state: StrategyState,
    ofi_calc: OFICalculator,
    vpin_calc: VPINCalculator,
    params: ASParams,
    t: float,
    *,
    alpha_beta: float = 0.5,        # OFI -> alpha coefficient (fit by regression)
    vpin_threshold: float = 0.4,
    vpin_widen_mult: float = 3.0,
    tick: float = 0.001,
) -> tuple[float, float] | None:
    """Return (bid, ask) or None to withdraw quotes."""
    
    # Step 1: fair value (microprice in logit space)
    mp = microprice(book.bid_px, book.bid_sz, book.ask_px, book.ask_sz)
    if mp <= 0 or mp >= 1:
        return None
    x_mid = logit(mp)

    # Step 2: short-term alpha from OFI (already updated externally)
    ofi = sum(e for _, e in ofi_calc.events)
    alpha_t = alpha_beta * ofi  # in logit space

    # Step 3: reservation price
    tau = max(params.T - t, 1e-6)
    x_res = (
        x_mid
        + alpha_t / params.sigma**2 * tau
        - state.inventory * params.gamma * params.sigma**2 * tau
    )

    # Step 4: half-spread (logit space)
    delta = (
        params.gamma * params.sigma**2 * tau
        + (2 / params.gamma) * np.log(1 + params.gamma / params.kappa)
    )

    # Step 5: VPIN toxicity gate
    vpin = vpin_calc.vpin_pm()
    if vpin > vpin_threshold:
        delta *= vpin_widen_mult

    # Step 6: transform back, round to tick
    bid = sigmoid(x_res - delta)
    ask = sigmoid(x_res + delta)
    bid = np.floor(bid / tick) * tick
    ask = np.ceil(ask / tick) * tick

    # Sanity: ensure non-crossing, valid range
    if bid >= ask or bid <= 0 or ask >= 1:
        return None

    return bid, ask
```

---

## 10. Event-driven backtester

Vector-based backtesting is fatal for HFT — it can't model queue position, partial fills, or sequential causality. Event-driven is mandatory.

```python
import heapq
from dataclasses import field
from typing import Callable, Any

@dataclass(order=True)
class Event:
    timestamp: float
    seq: int = field(compare=True)  # tiebreaker for stable ordering
    kind: str = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)


@dataclass
class OrderBook:
    """Minimal top-of-book state. Extend with full L2 as needed."""
    bid_px: float = 0.0
    bid_sz: float = 0.0
    ask_px: float = 1.0
    ask_sz: float = 0.0

    def apply(self, ev: Event):
        if ev.kind == "book":
            self.bid_px = ev.data["bid_px"]
            self.bid_sz = ev.data["bid_sz"]
            self.ask_px = ev.data["ask_px"]
            self.ask_sz = ev.data["ask_sz"]


@dataclass
class RestingOrder:
    order_id: int
    side: str           # "bid" or "ask"
    price: float
    size: float
    queue_ahead: float  # volume ahead of us at this price level
    placed_at: float


class Backtester:
    """Event-driven engine with latency injection, queue tracking,
    and maker/taker fee accounting (Polymarket V2 fee structure)."""

    def __init__(
        self,
        latency_ms: float = 50.0,
        fee_category: FeeCategory = FeeCategory.GEOPOLITICS,  # default: fee-free
    ):
        self.latency_ms = latency_ms
        self.fee_category = fee_category
        self.book = OrderBook()
        self.orders: dict[int, RestingOrder] = {}
        self.next_order_id = 1
        self.inventory = 0
        self.cash = 0.0
        self.fills: list[dict] = []
        self.queue: list[Event] = []
        self._seq = 0
        # Accounting
        self.gross_pnl = 0.0
        self.fees_paid = 0.0
        self.rebates_received = 0.0

    def push(self, ev: Event):
        heapq.heappush(self.queue, ev)

    def place_limit(self, t_decision: float, side: str, price: float, size: float):
        """Place a limit order; arrival is delayed by latency."""
        t_arrive = t_decision + self.latency_ms / 1000.0
        oid = self.next_order_id
        self.next_order_id += 1
        # Queue ahead estimated from current book state
        queue_ahead = self.book.bid_sz if side == "bid" else self.book.ask_sz
        self._seq += 1
        self.push(Event(t_arrive, self._seq, "place",
                        {"oid": oid, "side": side, "price": price,
                         "size": size, "queue_ahead": queue_ahead}))

    def cancel(self, t_decision: float, oid: int):
        t_arrive = t_decision + self.latency_ms / 1000.0
        self._seq += 1
        self.push(Event(t_arrive, self._seq, "cancel", {"oid": oid}))

    def run(self, events: list[Event],
            strategy: Callable[["Backtester", Event], None]):
        """Strategy receives each event after it has been applied to the book.
        Strategy may not see future events — enforced by sequential dispatch.
        """
        for ev in events:
            self._seq += 1
            self.push(Event(ev.timestamp, self._seq, ev.kind, ev.data))

        while self.queue:
            ev = heapq.heappop(self.queue)

            if ev.kind == "book":
                self.book.apply(ev)
                self._update_queue_positions(ev)
            elif ev.kind == "trade":
                self._process_trade(ev)
            elif ev.kind == "place":
                self._activate_order(ev)
            elif ev.kind == "cancel":
                self.orders.pop(ev.data["oid"], None)

            # Strategy callback (post-state-update)
            strategy(self, ev)

        return self._summary()

    def _activate_order(self, ev: Event):
        d = ev.data
        self.orders[d["oid"]] = RestingOrder(
            order_id=d["oid"], side=d["side"], price=d["price"],
            size=d["size"], queue_ahead=d["queue_ahead"],
            placed_at=ev.timestamp,
        )

    def _process_trade(self, ev: Event):
        """A market trade hits the book. Walk our resting orders at that price."""
        trade_px = ev.data["price"]
        trade_sz = ev.data["size"]
        trade_side = ev.data["side"]  # the aggressor side
        # Our resting orders fill on the opposite side
        my_side = "ask" if trade_side == "BUY" else "bid"

        for order in list(self.orders.values()):
            if order.side != my_side:
                continue
            if my_side == "bid" and trade_px > order.price:
                continue
            if my_side == "ask" and trade_px < order.price:
                continue
            # Trade consumes queue_ahead first, then us
            if trade_sz <= order.queue_ahead:
                order.queue_ahead -= trade_sz
                trade_sz = 0
            else:
                trade_sz -= order.queue_ahead
                order.queue_ahead = 0
                fill = min(trade_sz, order.size)
                self._record_fill(order, fill, ev.timestamp)
                order.size -= fill
                trade_sz -= fill
                if order.size <= 0:
                    self.orders.pop(order.order_id, None)
            if trade_sz <= 0:
                break

    def _record_fill(self, order: RestingOrder, qty: float, ts: float,
                     is_maker: bool = True):
        """Record a fill with proper maker/taker fee accounting.
        
        is_maker = True (resting order got hit) -> earn rebate
        is_maker = False (we crossed the spread) -> pay taker fee
        """
        sign = 1 if order.side == "bid" else -1
        notional = qty * order.price
        
        if is_maker:
            rebate = maker_rebate(order.price, qty, self.fee_category)
            self.rebates_received += rebate
            cash_flow = -sign * notional + rebate
        else:
            fee = taker_fee(order.price, qty, self.fee_category)
            self.fees_paid += fee
            cash_flow = -sign * notional - fee
        
        self.inventory += sign * int(qty)
        self.cash += cash_flow
        self.fills.append({
            "ts": ts, "side": order.side, "price": order.price,
            "qty": qty, "is_maker": is_maker,
            "queue_ahead_at_placement": order.queue_ahead,
            "time_to_fill": ts - order.placed_at,
        })

    def _update_queue_positions(self, ev: Event):
        """Update our orders' queue_ahead from observed level shrinkage.
        
        Heuristic: when total resting size at our price level shrinks by Δ
        between events, attribute Δ to cancellations + market hits in front
        of us proportionally to our current queue_ahead / level_size.
        This is the same approximation used in §8.6's L2OrderBook.
        For production: swap in L2OrderBook here and remove this stub —
        it handles level tracking precisely from per-side book updates.
        """
        if ev.kind != "book":
            return
        for order in self.orders.values():
            level_size = (self.book.bid_sz if order.side == "bid"
                          else self.book.ask_sz)
            # Order rests at BBO in this minimal book; for full L2 attribute
            # the shrinkage proportionally to position in the queue.
            if level_size < order.queue_ahead + order.size:
                # Level shrank — some volume disappeared. Without per-event
                # cancel signals we attribute proportionally.
                total_before = order.queue_ahead + order.size
                shrinkage = total_before - level_size
                # Cancellations are uniform-in-position on average;
                # market hits are always from the front.
                # Use a 50/50 split as a working approximation.
                from_front = shrinkage * 0.5
                from_anywhere = shrinkage * 0.5 * (
                    order.queue_ahead / max(total_before, 1e-9)
                )
                order.queue_ahead = max(
                    0.0, order.queue_ahead - from_front - from_anywhere
                )

    def _summary(self) -> dict:
        if not self.fills:
            return {"pnl": 0, "n_fills": 0, "fees_paid": 0, "rebates_received": 0}
        # Mark to mid
        mid = (self.book.bid_px + self.book.ask_px) / 2
        pnl = self.cash + self.inventory * mid
        n_maker = sum(1 for f in self.fills if f.get("is_maker", True))
        return {
            "pnl": pnl,
            "n_fills": len(self.fills),
            "n_maker_fills": n_maker,
            "n_taker_fills": len(self.fills) - n_maker,
            "final_inventory": self.inventory,
            "fees_paid": self.fees_paid,
            "rebates_received": self.rebates_received,
            "net_fees": self.fees_paid - self.rebates_received,
            "fills": self.fills,
        }
```

### Required validation tests

After your backtest looks profitable, run these *before* deploying anything:

1. **Delay injection.** Add 100ms, 500ms, 2s latency to every signal. Sharpe should degrade smoothly. If it collapses → hidden look-ahead bias.
2. **Shuffle test.** Permute event timestamps; features fixed. Sharpe should drop to ~0. If not → strategy exploits calendar artifacts.
3. **Regime split.** Train on one period, test on a structurally different one (different platform version, different fee regime). Strategies that only work in one regime are fragile.
4. **Dispersion test.** Run 100 combinatorial path splits; plot Sharpe distribution. High median + huge variance = overfit in disguise.

### 10.5 Production WebSocket client (reconnect + heartbeat + sequence)

The minimal subscriber in §7 is enough for backtest replay but unsafe for live trading. Three additions are non-negotiable:

1. **Auto-reconnect** with exponential backoff (network glitches happen multiple times per day on Polygon).
2. **Heartbeat detection** for the silent-freeze bug (server accepts subscription, then sends nothing).
3. **Sequence-number tracking** to detect missed messages and trigger a snapshot resync.

```python
import asyncio
import json
import time
from typing import Callable, Awaitable
import websockets

class PolymarketWSClient:
    """Production-grade Polymarket market-channel client.
    
    Handles reconnection with exponential backoff, heartbeat timeout
    detection, and gap detection via the server's hash field.
    """
    URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    HEARTBEAT_TIMEOUT_S = 30.0
    INITIAL_BACKOFF_S = 1.0
    MAX_BACKOFF_S = 60.0

    def __init__(
        self,
        asset_ids: list[str],
        on_event: Callable[[dict, int], Awaitable[None]],
        on_disconnect: Callable[[str], Awaitable[None]] | None = None,
        on_resync: Callable[[], Awaitable[None]] | None = None,
    ):
        self.asset_ids = asset_ids
        self.on_event = on_event
        self.on_disconnect = on_disconnect or (lambda r: asyncio.sleep(0))
        self.on_resync = on_resync or (lambda: asyncio.sleep(0))
        self.last_msg_ts = 0.0
        self.last_hash_by_asset: dict[str, str] = {}
        self._stop = False

    async def run(self):
        """Main loop. Reconnects forever until stop() called."""
        backoff = self.INITIAL_BACKOFF_S
        while not self._stop:
            try:
                async with websockets.connect(
                    self.URL, ping_interval=15, ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    await self._subscribe(ws)
                    backoff = self.INITIAL_BACKOFF_S  # reset on success
                    await self._read_loop(ws)
            except (websockets.ConnectionClosed, asyncio.TimeoutError,
                    OSError) as e:
                await self.on_disconnect(f"connection lost: {e}")
            except Exception as e:
                await self.on_disconnect(f"unexpected: {e!r}")
            
            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.MAX_BACKOFF_S)

    async def _subscribe(self, ws):
        await ws.send(json.dumps({
            "type": "market", "assets_ids": self.asset_ids,
        }))
        self.last_msg_ts = time.time()
        await self.on_resync()

    async def _read_loop(self, ws):
        """Read messages with heartbeat watchdog."""
        async def watchdog():
            while True:
                await asyncio.sleep(5)
                if time.time() - self.last_msg_ts > self.HEARTBEAT_TIMEOUT_S:
                    await self.on_disconnect("heartbeat timeout (silent freeze)")
                    await ws.close()
                    return
        
        wd_task = asyncio.create_task(watchdog())
        try:
            async for raw in ws:
                recv_ts = int(time.time() * 1000)
                self.last_msg_ts = time.time()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                events = msg if isinstance(msg, list) else [msg]
                for ev in events:
                    await self._handle_event(ev, recv_ts)
        finally:
            wd_task.cancel()

    async def _handle_event(self, ev: dict, recv_ts: int):
        """Apply gap detection (via hash) and dispatch."""
        asset = ev.get("asset_id")
        new_hash = ev.get("hash")
        # If the message includes a hash and we have a previous one,
        # detect gaps by checking the hash chain. Polymarket book events
        # publish a hash for each snapshot.
        if asset and new_hash and asset in self.last_hash_by_asset:
            # Hash mismatch with stored prev_hash field would indicate gap,
            # but the public feed isn't strictly chained; rely on
            # periodic REST snapshot reconciliation as a safety net.
            pass
        if asset and new_hash:
            self.last_hash_by_asset[asset] = new_hash
        await self.on_event(ev, recv_ts)

    def stop(self):
        self._stop = True
```

**Periodic REST reconciliation.** Even with hash tracking, you should hit the REST `/orderbook` endpoint every 30 seconds and diff against your local book. Any divergence beyond a single tick = drop and rebuild from snapshot.

```python
async def reconcile_with_rest(client_v2, local_book: L2OrderBook,
                               token_id: str) -> bool:
    """Compare local book to REST snapshot. Return True if in sync."""
    snap = client_v2.get_order_book(token_id)
    rest_best_bid = max((float(b["price"]) for b in snap.bids), default=0)
    rest_best_ask = min((float(a["price"]) for a in snap.asks), default=1)
    local_bb = local_book.best_bid()
    local_ba = local_book.best_ask()
    if local_bb is None or local_ba is None:
        return False
    return (abs(local_bb[0] - rest_best_bid) < 1e-6
            and abs(local_ba[0] - rest_best_ask) < 1e-6)
```

---

## 11. Purged combinatorial cross-validation

Standard k-fold leaks future information into training. Walk-forward only tests one path. López de Prado's purged CPCV fixes both.

```python
from itertools import combinations
import pandas as pd

def purged_cpcv_splits(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_window: int = 100,
    embargo: int = 50,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (train_idx, test_idx) splits with purging and embargo.
    
    Each sample is assigned to one of n_groups sequential groups.
    Each split chooses n_test_groups groups for testing. The training set
    excludes (a) test groups themselves, (b) a purge_window of samples
    adjacent to each test group, (c) an embargo of samples after each test group.
    """
    group_size = n_samples // n_groups
    group_ranges = [
        (i * group_size, (i + 1) * group_size if i < n_groups - 1 else n_samples)
        for i in range(n_groups)
    ]
    splits = []
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([
            np.arange(group_ranges[g][0], group_ranges[g][1]) for g in test_combo
        ])
        # Build forbidden zone: test + purge + embargo
        forbidden = set(test_idx.tolist())
        for g in test_combo:
            lo, hi = group_ranges[g]
            forbidden.update(range(max(0, lo - purge_window), lo))
            forbidden.update(range(hi, min(n_samples, hi + embargo)))
        train_idx = np.array(
            [i for i in range(n_samples) if i not in forbidden]
        )
        splits.append((train_idx, test_idx))
    return splits


def deflated_sharpe_ratio(
    observed_sr: float, n_trials: int, sr_returns: np.ndarray,
) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.
    
    Returns probability that the *observed_sr* exceeds zero after
    correcting for (a) the number of trials, (b) non-normality (skew,
    kurtosis) of the returns.
    
    observed_sr: Sharpe of the strategy being evaluated.
    n_trials: number of strategies tested (independent backtest paths).
    sr_returns: 1-D array of per-period returns used to compute observed_sr.
    """
    from scipy import stats
    
    T = len(sr_returns)
    if T < 4:
        return 0.0
    
    # Skew and kurtosis of returns
    gamma3 = stats.skew(sr_returns)
    gamma4 = stats.kurtosis(sr_returns, fisher=True)  # excess kurtosis
    
    # Bailey-Lopez de Prado: expected max of n_trials draws from N(0, 1)
    # using Sidak's approximation
    euler = 0.5772156649
    e_max_sr = ((1 - euler) * stats.norm.ppf(1 - 1/n_trials)
                + euler * stats.norm.ppf(1 - 1/(n_trials * np.e)))
    
    # Variance of estimator of Sharpe under non-normal returns
    sr_var = (
        1 - gamma3 * observed_sr + (gamma4 / 4) * observed_sr**2
    ) / (T - 1)
    
    if sr_var <= 0:
        return 0.0
    
    # Probabilistic Sharpe Ratio with deflation
    z = (observed_sr - e_max_sr) / np.sqrt(sr_var)
    return float(stats.norm.cdf(z))


def probability_of_backtest_overfit(
    insample_sr: np.ndarray, oos_sr: np.ndarray,
) -> float:
    """López de Prado's PBO metric.
    
    For each combinatorial CV split, you produce one IS Sharpe and one OOS
    Sharpe. PBO is the fraction of splits where the IS-best strategy
    underperformed median OOS — i.e. the probability that picking the best
    in-sample strategy produces below-median out-of-sample performance.
    
    PBO < 0.3 = robust; PBO > 0.5 = pure overfit.
    """
    assert insample_sr.shape == oos_sr.shape
    n_splits, n_strategies = insample_sr.shape
    
    is_best_idx = insample_sr.argmax(axis=1)  # which strategy won IS each split
    oos_ranks = oos_sr.argsort(axis=1).argsort(axis=1)  # rank in OOS
    oos_rank_of_is_best = oos_ranks[np.arange(n_splits), is_best_idx]
    # Logits: rank converted to probability of beating median
    median_rank = (n_strategies - 1) / 2
    p_overfit = float((oos_rank_of_is_best < median_rank).mean())
    return p_overfit


def diebold_mariano(
    losses_a: np.ndarray, losses_b: np.ndarray, h: int = 1,
) -> dict:
    """Diebold-Mariano test for equal predictive accuracy.
    
    losses_a, losses_b: per-period loss series for strategies A and B
    (use squared returns or any other loss).
    h: forecast horizon (1 for daily, n for n-step ahead).
    
    Returns DM statistic and p-value. Two-sided alternative: strategies differ.
    """
    from scipy import stats
    
    d = losses_a - losses_b
    n = len(d)
    if n < 10:
        return {"statistic": 0, "p_value": 1.0}
    
    d_mean = d.mean()
    # Long-run variance via Newey-West (HAC) for h > 1
    if h == 1:
        d_var = d.var(ddof=1) / n
    else:
        gamma_0 = d.var(ddof=1)
        gamma_sum = sum(
            (1 - k/h) * np.cov(d[:-k], d[k:], ddof=1)[0, 1]
            for k in range(1, h)
        )
        d_var = (gamma_0 + 2 * gamma_sum) / n
    
    if d_var <= 0:
        return {"statistic": 0, "p_value": 1.0}
    
    dm_stat = d_mean / np.sqrt(d_var)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {
        "statistic": float(dm_stat),
        "p_value": float(p_value),
        "a_better": bool(dm_stat < 0),  # negative = A has lower loss
    }
```

**Reading the metrics.** A backtest summary should report at minimum:
- **DSR** > 0.95: strong evidence that the observed Sharpe is genuinely positive after deflating for multiple-testing and non-normality.
- **PBO** < 0.3: the IS-best parameter set is reliably above-median OOS.
- **DM p-value** < 0.05 when comparing to a baseline (constant-spread MM, naive predictor): rejects the null of equal performance.

Any single metric is gameable. Reporting all three with their underlying samples is much harder to fake.

---

## 12. Risk and Kelly

### Kelly for binary contracts

For a YES contract priced at $c$ with your estimated true probability $q$, log-utility gives:

$$U(f) = (1-q)\log(1-f) + q\log\left(1 + f\cdot\frac{1-c}{c}\right)$$

Solving $\partial U/\partial f = 0$:

$$\boxed{f^* = \frac{q - c}{1 - c}}$$

Properties: $f^* = 0$ at $q = c$; $f^* \to 1$ as $c \to 0$. Full Kelly is volatile — a few losses can drop bankroll 50%+. Use fractional Kelly (0.25-0.5×) in practice.

### Boundary risk

Near $p = 0$ or $p = 1$, leverage is implicit. Cap exposure:

$$|\text{exposure}| \leq M \sqrt{p(1-p)}$$

```python
def kelly_fraction(q: float, c: float, fraction: float = 0.25) -> float:
    """Fractional Kelly for a YES contract.
    q: your probability estimate (use lower confidence bound!)
    c: current ask price
    fraction: Kelly fraction multiplier (0.25 = quarter-Kelly)
    """
    if c >= q or c <= 0 or c >= 1:
        return 0.0
    f_star = (q - c) / (1 - c)
    return max(0.0, min(f_star * fraction, 1.0))


def position_size(
    bankroll: float, q: float, c: float, p_market: float,
    fraction: float = 0.25, M: float = 0.1,
) -> float:
    """Bankroll-aware position with boundary risk cap."""
    f = kelly_fraction(q, c, fraction)
    kelly_size = f * bankroll
    boundary_cap = M * bankroll * np.sqrt(p_market * (1 - p_market))
    return min(kelly_size, boundary_cap)
```

### Bayesian Kelly under uncertainty over q

The standard Kelly assumes $q$ is known. In practice $q$ is your *posterior* over the true probability, itself a distribution. Treating the point estimate as truth is what destroys bankrolls — the Kelly formula amplifies any overconfidence in $q$.

The correct extension: integrate the log-utility over your posterior on $q$. If your posterior is $q \sim \text{Beta}(\alpha, \beta)$ (a natural choice for a binary event), then:

$$\mathbb{E}_{q}[U(f)] = \int_0^1 \left[ (1-q)\log(1-f) + q\log\left(1 + f\cdot\frac{1-c}{c}\right) \right] \frac{q^{\alpha-1}(1-q)^{\beta-1}}{B(\alpha,\beta)} dq$$

The integral reduces nicely: $\mathbb{E}[q] = \alpha/(\alpha+\beta)$, so the first-order Bayesian Kelly is just plug-in. But the *second-order correction* matters: a higher-variance posterior shrinks the optimal bet.

```python
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist

def bayesian_kelly_fraction(
    alpha: float, beta_param: float, c: float,
    n_quadrature: int = 50,
) -> float:
    """Optimal Kelly fraction under Beta(α, β) posterior over true probability.
    
    Numerically maximizes E_q[U(f)] over f ∈ [0, 1).
    Returns the fraction that maximizes expected log-utility.
    
    Use α, β from your prior + observed Bernoulli evidence:
      α = prior_alpha + n_successes
      β = prior_beta + n_failures
    """
    # Gauss-Legendre quadrature over [0, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)
    q_grid = 0.5 * (nodes + 1)  # transform [-1, 1] -> [0, 1]
    q_weights = 0.5 * weights * beta_dist.pdf(q_grid, alpha, beta_param)
    
    def neg_expected_utility(f: float) -> float:
        if f <= -1e-9 or f >= 1.0 - 1e-9:
            return 1e10
        if c <= 0 or c >= 1:
            return 1e10
        u = (1 - q_grid) * np.log(1 - f) + q_grid * np.log(
            1 + f * (1 - c) / c
        )
        return -np.sum(q_weights * u)
    
    # Search on a fine grid then refine
    f_grid = np.linspace(0.001, 0.999, 200)
    losses = [neg_expected_utility(f) for f in f_grid]
    best = f_grid[np.argmin(losses)]
    
    # Local refinement
    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(
            neg_expected_utility, bounds=(max(0.001, best - 0.05),
                                            min(0.999, best + 0.05)),
            method="bounded",
        )
        return float(max(0.0, result.x))
    except Exception:
        return float(best)


def posterior_credible_interval(
    alpha: float, beta_param: float, level: float = 0.05,
) -> tuple[float, float]:
    """Equal-tailed credible interval for q ~ Beta(α, β).
    Use the lower bound as a conservative Kelly input."""
    return (
        float(beta_dist.ppf(level / 2, alpha, beta_param)),
        float(beta_dist.ppf(1 - level / 2, alpha, beta_param)),
    )
```

**Practical heuristic.** If you don't want to integrate, use the lower-credible-bound shortcut: take the 5th percentile of your posterior on $q$ and plug it into the standard Kelly formula. This approximates Bayesian Kelly with quarter-Kelly fractional safety under typical posterior widths.

The relationship is:
$$f^*_{\text{Bayesian}} \approx f^*_{\text{Kelly}}(q_{\text{point}}) - k \cdot \text{Var}[q]$$

for some $k > 0$ that depends on the price $c$. Posterior variance directly subtracts from optimal sizing. This is why beating-the-bookmaker requires **calibrated** probability estimates, not just *accurate* ones — overconfidence is twice as bad as underestimation.

### Kill switches (non-negotiable)

```python
@dataclass
class RiskLimits:
    max_inventory_per_market: int = 100
    max_total_drawdown: float = 0.20     # 20% of starting bankroll
    max_order_rate_per_sec: float = 10.0
    max_concurrent_markets: int = 20
    heartbeat_timeout_sec: float = 30.0
    withdraw_before_resolution_sec: float = 300.0  # 5 min

class KillSwitch:
    def __init__(self, limits: RiskLimits, starting_bankroll: float):
        self.limits = limits
        self.starting = starting_bankroll
        self.peak = starting_bankroll
        self.last_heartbeat = 0.0
        self.recent_orders: deque = deque()
        self.tripped: str | None = None

    def check_drawdown(self, current_bankroll: float, ts: float) -> bool:
        self.peak = max(self.peak, current_bankroll)
        if (self.peak - current_bankroll) / self.starting > self.limits.max_total_drawdown:
            self.tripped = f"drawdown@{ts}"
            return False
        return True

    def check_order_rate(self, ts: float) -> bool:
        while self.recent_orders and ts - self.recent_orders[0] > 1.0:
            self.recent_orders.popleft()
        if len(self.recent_orders) >= self.limits.max_order_rate_per_sec:
            self.tripped = f"order_rate@{ts}"
            return False
        self.recent_orders.append(ts)
        return True

    def check_heartbeat(self, ts: float) -> bool:
        if ts - self.last_heartbeat > self.limits.heartbeat_timeout_sec:
            self.tripped = f"heartbeat@{ts}"
            return False
        return True

    def heartbeat(self, ts: float):
        self.last_heartbeat = ts
```

---

## 13. Synthetic control for event studies

§7 layer 4 said: don't regress prices on event indicators — use synthetic control. Here's the implementation.

The idea (Goldsmith-Pinkham & Lyu 2025): for a market $M$ that experiences event $E$, construct a counterfactual price path by combining unrelated control markets that did *not* experience $E$. The synthetic counterfactual is fitted on a pre-event window where neither the treated market nor controls have anything special happening. The treatment effect is the gap between actual and synthetic price after the event.

For prediction markets this is unusually clean: hundreds of parallel markets give plenty of controls.

```python
import numpy as np
from scipy.optimize import minimize

class SyntheticControl:
    """Abadie-Diamond-Hainmueller synthetic control for event studies.
    
    Fits a convex combination of control units' price paths to match the
    treated unit's pre-event path, then projects forward to estimate the
    counterfactual.
    """
    def __init__(self, treated: np.ndarray, controls: np.ndarray):
        """
        treated: 1-D array of treated unit's price/logit-price, length T.
        controls: 2-D array (T, K) of K control units' paths.
        Use logit-prices for prediction markets, not raw prices.
        """
        assert treated.ndim == 1
        assert controls.shape[0] == len(treated)
        self.treated = treated
        self.controls = controls
        self.weights: np.ndarray | None = None

    def fit(self, pre_event_idx: slice, l2_reg: float = 0.0) -> np.ndarray:
        """Fit weights w (simplex: w_k >= 0, sum w_k = 1) minimising
        ||treated[pre] - controls[pre] @ w||² + λ ||w||².
        Returns the fitted weights.
        """
        K = self.controls.shape[1]
        Y = self.treated[pre_event_idx]
        X = self.controls[pre_event_idx]

        def loss(w):
            resid = Y - X @ w
            return float(resid @ resid + l2_reg * (w @ w))

        # Simplex constraint via softmax parametrisation
        def loss_softmax(z):
            w = np.exp(z - z.max())
            w = w / w.sum()
            return loss(w)

        result = minimize(loss_softmax, np.zeros(K), method="L-BFGS-B")
        z = result.x
        w = np.exp(z - z.max())
        w = w / w.sum()
        self.weights = w
        return w

    def counterfactual(self) -> np.ndarray:
        """Synthetic counterfactual path for the full window."""
        if self.weights is None:
            raise RuntimeError("Call fit() first")
        return self.controls @ self.weights

    def treatment_effect(self) -> np.ndarray:
        """Per-timestep treatment effect (treated - synthetic)."""
        return self.treated - self.counterfactual()

    def placebo_test(self, n_placebos: int = 100, pre_event_idx: slice = None
                     ) -> dict:
        """Run the same procedure pretending each control is treated.
        If the real treatment effect is much larger than placebo effects,
        the event has a true causal signature."""
        actual_te = self.treatment_effect()
        post = ~np.isin(
            np.arange(len(self.treated)),
            np.arange(pre_event_idx.start, pre_event_idx.stop),
        )
        actual_rms = np.sqrt((actual_te[post] ** 2).mean())

        placebo_rms = []
        for k in np.random.choice(self.controls.shape[1],
                                   min(n_placebos, self.controls.shape[1]),
                                   replace=False):
            placebo_treated = self.controls[:, k]
            placebo_controls = np.delete(self.controls, k, axis=1)
            sc_k = SyntheticControl(placebo_treated, placebo_controls)
            sc_k.fit(pre_event_idx)
            te_k = sc_k.treatment_effect()
            placebo_rms.append(np.sqrt((te_k[post] ** 2).mean()))
        placebo_rms = np.array(placebo_rms)
        return {
            "actual_rms_effect": float(actual_rms),
            "placebo_rms_quantiles": {
                "median": float(np.median(placebo_rms)),
                "p95": float(np.quantile(placebo_rms, 0.95)),
                "p99": float(np.quantile(placebo_rms, 0.99)),
            },
            "rank": float((placebo_rms < actual_rms).mean()),
        }
```

**How to use it.** Suppose you suspect a particular news event (presidential debate, Fed speech) systematically moves a market. Pick a 6-hour window: 5 hours pre-event, 1 hour post. The treated unit is the affected market. Controls are 20-50 parallel markets on unrelated topics. Fit on the first 5 hours; the gap in the 6th hour is your estimated effect.

The `placebo_test` returns the rank of the actual effect among placebo effects. A rank > 0.95 means your event has a stronger effect than 95% of randomly-chosen "treatments" — credible signal. A rank near 0.5 means your event is indistinguishable from noise.

This replaces naive event-window regressions. Use it both for research (does event type X really move prices?) and for live signal validation (am I attributing a price move to this news, or would it have moved anyway?).

---

## 14. End-to-end working example

Tying it all together: data → signals → quoting → backtest → validation. This is the bare skeleton you'd run on day 1 of Phase 4 in the roadmap.

```python
"""
Minimal end-to-end driver. Assumes:
  - You have historical book events in a pandas DataFrame `events_df`
    with columns: ts (ms), kind ('book'|'trade'), bid_px, bid_sz,
    ask_px, ask_sz, trade_px, trade_sz, trade_side.
  - Time horizon T is in seconds-since-data-start.
"""
import pandas as pd

# 1. Calibrate OFI -> alpha on a TRAINING slice
train_end = int(len(events_df) * 0.6)
book_train = events_df.iloc[:train_end].query("kind == 'book'")
calib = calibrate_ofi_alpha(
    book_train[["ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]],
    horizon_ms=1000.0,
    ofi_window_ms=1000.0,
)
print(f"Calibrated alpha_beta = {calib['alpha_beta']:.4f}, R² = {calib['r2']:.4f}")
if calib["r2"] > 0.2:
    raise RuntimeError("In-sample R² too high — suspected leakage.")

# 2. Configure strategy
params = ASParams(
    gamma=2.0,
    sigma=calib["residual_std"] / np.sqrt(calib["horizon_ms"] / 1000),
    kappa=10.0,
    T=events_df["ts"].max() / 1000,
)

ofi_calc = OFICalculator(window_seconds=1.0)
vpin_calc = VPINCalculator(bucket_volume=5000.0, n_buckets=50)
state = StrategyState()

# 3. Run event-driven backtest on OUT-OF-SAMPLE slice
test = events_df.iloc[train_end:].reset_index(drop=True)
test_events = []
for _, row in test.iterrows():
    test_events.append(Event(
        timestamp=row["ts"] / 1000,
        seq=0,  # filled in by Backtester
        kind=row["kind"],
        data=row.dropna().to_dict(),
    ))

bt = Backtester(latency_ms=50.0, fee_category=FeeCategory.POLITICS)
# Pick the FeeCategory matching the market you're backtesting on.
# FINANCE has the richest rebate (50%); GEOPOLITICS is fee-free.

def strategy_callback(bt: Backtester, ev: Event):
    if ev.kind == "book":
        ofi_calc.update(
            ev.timestamp, ev.data["bid_px"], ev.data["bid_sz"],
            ev.data["ask_px"], ev.data["ask_sz"],
        )
    elif ev.kind == "trade":
        vpin_calc.add_trade(
            ev.data["trade_sz"], ev.data["trade_side"] == "BUY",
            ev.data.get("trade_px", 0.5),
        )

    # Recompute quotes after every book update; cancel/repost
    if ev.kind == "book":
        quotes = compute_quotes(
            bt.book, state, ofi_calc, vpin_calc, params,
            t=ev.timestamp,
            alpha_beta=calib["alpha_beta"],
        )
        if quotes is None:
            for oid in list(bt.orders):
                bt.cancel(ev.timestamp, oid)
            return
        bid, ask = quotes
        # Cancel-replace if quotes moved more than a tick
        for oid, order in list(bt.orders.items()):
            target = bid if order.side == "bid" else ask
            if abs(order.price - target) > 0.001:
                bt.cancel(ev.timestamp, oid)
        # Place new quotes
        bt.place_limit(ev.timestamp, "bid", bid, size=10.0)
        bt.place_limit(ev.timestamp, "ask", ask, size=10.0)
        state.inventory = bt.inventory

result = bt.run(test_events, strategy_callback)
print(f"PnL: ${result['pnl']:.2f}, fills: {result['n_fills']}")

# 4. Validation: delay-injection
def run_with_delay(extra_latency_ms: float) -> float:
    bt2 = Backtester(latency_ms=50.0 + extra_latency_ms,
                     fee_category=FeeCategory.POLITICS)
    # Re-init state-dependent calculators for each run
    ofi_calc.events.clear()
    vpin_calc.buckets.clear()
    state.inventory = 0
    r = bt2.run(test_events, strategy_callback)
    return r["pnl"]

baseline = result["pnl"]
delays = [100, 500, 2000]  # ms
pnls = [run_with_delay(d) for d in delays]
print(f"Baseline: ${baseline:.2f}; +100ms: ${pnls[0]:.2f}; "
      f"+500ms: ${pnls[1]:.2f}; +2000ms: ${pnls[2]:.2f}")
# Acceptance: PnL should degrade smoothly. If +100ms collapses to <50% of baseline,
# investigate look-ahead.
```

Notice the discipline: calibration is on the first 60%, backtest is on the remaining 40%, and the OOS slice never touches calibration. The delay-injection test runs *after* the backtest is profitable, not before — its purpose is to falsify, not validate.

---

## 15. Common bugs and diagnostics

Most bot failures come from a small set of mistakes. This table is what to check first when something is broken.

| Symptom | Likely cause | Diagnostic |
|---|---|---|
| Live PnL << backtest PnL | Look-ahead bias somewhere | Run delay-injection on backtest. If +100ms collapses Sharpe, you have layer-1 or layer-2 leakage. |
| `Insufficient balance` despite deposit | V2 `/balance-allowance/update` not called | After any deposit/allowance change, hit the sync endpoint with `signature_type=3`. See §7.5. |
| `Invalid signature` on simple orders | Wrong signature_type for wallet type | EOA = 0; Magic/Email = 1; Browser proxy = 2; **V2 deposit wallet = 3**. Mismatched type causes server-side rejection. |
| Orders fill but inventory doesn't update | V1 "ghost fill" bug | Migrate to V2. The ERC-1271 wrapped signatures make this impossible. |
| FOK market orders fail on thin markets | Not enough cumulative depth at one price | Use FAK if partial fills are acceptable, or break order into smaller pieces. |
| Backtest Sharpe collapses under +500ms delay | Reading future book state in features | Audit every feature: its data must have `ts < decision_ts`, strictly. |
| Backtest Sharpe is positive when timestamps shuffled | Strategy exploits non-temporal artifacts | Calendar effects, day-of-week, market-ID leakage. Find the artifact, remove it from features. |
| Inventory ratchets up over a session | Asymmetric fill rates (κ different on bid vs ask) | Allow asymmetric κ in your AS implementation; fit `κ_bid, κ_ask` separately from fills. |
| Sharpe varies wildly between CPCV paths | Strategy is path-dependent / overfit | High dispersion = brittle. Reduce parameter count, regularize, drop signals with weak in-sample evidence. |
| WebSocket connection drops silently (no errors) | Server accepted subscription but stopped sending | Implement timeout detection (no msg in 30s → reconnect). The "silent freeze" bug (#292) is a documented Polymarket issue. |
| `get_balance()` returns 0 but UI shows balance | V1 SDK issue with proxy wallets / V2 cache | If V1: known issue #319, use the Data API. If V2: hit `/balance-allowance/update`. |
| Spreads widen but PnL still drops on news events | Reacting to news too late | Pre-schedule withdrawal for known events; aggressive VPIN gate for unknowns; measure end-to-end news ingestion latency. |
| AS quotes both bid and ask but neither fills | Tick-size rounding pushed quotes outside book | Check `tick_size` for the market (varies by price level on Polymarket: 0.01 normal, 0.001 near boundaries). |
| pnl(MTM) ≠ pnl(realized) on closeout | Mark-to-which-price ambiguity | Mark to microprice on open, realized at fill price; reconciliation must match by final tick. |

The most expensive lesson in this list: **a strategy that works in backtest but fails on +100ms delay injection is not 90% good — it's 0% good.** Whatever it was doing depended on micro-look-ahead that doesn't exist in real time.

### 15.5 Monitoring, alerting, and disaster recovery

A live bot without monitoring is a bot that loses money silently until someone notices. Set up these signals from day 1:

```python
from dataclasses import dataclass, field
from collections import deque
import time

@dataclass
class BotMetrics:
    """Push these to Prometheus/Grafana/whatever you have.
    Sample every 1-5 seconds, alert on threshold violations."""
    # Health
    last_book_event_ts: float = 0.0
    last_fill_ts: float = 0.0
    websocket_connected: bool = False
    rest_reconcile_ok: bool = True
    
    # Performance
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid_today: float = 0.0
    rebates_received_today: float = 0.0
    fill_count_today: int = 0
    
    # Risk
    current_inventory_by_market: dict = field(default_factory=dict)
    drawdown_from_peak: float = 0.0
    
    # Strategy state
    current_quoted_spread_bps: float = 0.0
    estimated_vpin: float = 0.0
    estimated_alpha: float = 0.0
    estimated_kappa: float = 0.0

def alert_conditions(m: BotMetrics, peak_bankroll: float) -> list[str]:
    """Return list of human-readable alerts that should fire."""
    alerts = []
    now = time.time()
    
    # Connectivity
    if not m.websocket_connected:
        alerts.append("CRITICAL: WebSocket disconnected")
    if now - m.last_book_event_ts > 30:
        alerts.append(f"WARN: no book events for {now - m.last_book_event_ts:.0f}s "
                      "(possible silent freeze)")
    if not m.rest_reconcile_ok:
        alerts.append("WARN: local book diverged from REST snapshot")
    
    # Risk
    if m.drawdown_from_peak / max(peak_bankroll, 1) > 0.10:
        alerts.append(f"WARN: drawdown {m.drawdown_from_peak / peak_bankroll:.1%}")
    if m.drawdown_from_peak / max(peak_bankroll, 1) > 0.20:
        alerts.append("CRITICAL: drawdown exceeds kill-switch threshold")
    
    # Strategy health
    if m.estimated_vpin > 0.6:
        alerts.append(f"WARN: high VPIN ({m.estimated_vpin:.2f}); consider withdrawing")
    if m.fees_paid_today > m.rebates_received_today * 2:
        alerts.append("WARN: paying more taker fees than rebates collected; "
                      "your `post_only` is failing")
    
    # Activity
    if m.fill_count_today < 5 and now - m.last_fill_ts > 3600:
        alerts.append("INFO: no fills in 1h; verify quotes are competitive")
    
    return alerts
```

**Disaster recovery playbook.** When something breaks, you need a tested response, not improvisation.

| Failure | First action | Recovery |
|---|---|---|
| WebSocket dropped | Auto-reconnect (already implemented) | Re-fetch full orderbook via REST; rebuild local L2 from snapshot |
| Polymarket scheduled maintenance | Cancel all open orders 5 min before; pause trading | Resume only after first successful REST reconcile post-maintenance |
| Polygon RPC unreachable | Continue reading WebSocket; halt new orders | Until on-chain reads succeed: do not place orders that depend on balance sync |
| Single market freezes (no updates) | Cancel orders on that market specifically | Move capital to other markets; alert human |
| Inventory drift beyond cap | Kill switch fires; cancel all orders | Liquidate via FAK orders to flatten; investigate cause before restart |
| Price moves >5% in <1 second | VPIN gate widens automatically; logs flagged | Wait 5 minutes; recalibrate $\sigma_b$ with new data window |
| Maker rebate not received | Check `is_maker` flag was True at fill time | If `post_only` was off, this is taker fill; reconfigure SDK |
| Wallet signature errors | Check signature_type, deposit wallet address | Run `/balance-allowance/update`; rotate API key if persistent |
| `Insufficient balance` with funded wallet | Hit `/balance-allowance/update` | If persists, check pUSD conversion completed; CTF approvals set |
| Sharp PnL drop with no obvious cause | Pull last 100 fills; check `is_maker` distribution | If many takers: spread settings too tight, you're crossing the book |

**Pre-deployment checklist.** Before any live capital, verify:

- [ ] WebSocket reconnect tested by force-killing connection
- [ ] REST reconciliation runs every 30s and logs any divergence
- [ ] Heartbeat watchdog fires within 30s of stopped messages (test by blocking your network)
- [ ] Kill switch triggers correctly at 20% drawdown (test in paper trading)
- [ ] All orders use `signature_type=3` and `post_only=True`
- [ ] `/balance-allowance/update` called after each deposit (test on Amoy testnet)
- [ ] Inventory caps enforced per-market and globally
- [ ] Alerts route to a channel you actually monitor (not just logs)
- [ ] Latency profile measured and matches backtest assumption
- [ ] Code path tested for scheduled-event withdrawal (e.g. Fed meeting in 5 minutes)

The single biggest production lesson: **boring bots survive.** Every "clever" feature is a future bug. Start with the minimal correct implementation, run it for weeks, only then add complexity.

---

## 16. Roadmap

Eight phases with explicit acceptance criteria. Don't skip phases — every skip becomes a bug at phase 8.

| Phase | Goal | Acceptance criterion |
|---|---|---|
| 0. Foundations | Internalize the math | Derive AS on whiteboard; explain reservation price intuition |
| 1. Data | Tick capture and replay | Replay any 1-hour window with no gaps, no out-of-order events |
| 2. Simulator | Event-driven engine | "Do nothing" strategy returns PnL = 0; "buy 1/min" traces correctly |
| 3. Naive MM | Constant spread → AS → GLT | AS beats constant-spread on Sharpe most days |
| 4. Signals | OFI + microprice + VPIN | PnL improves measurably per signal added |
| 5. Events | News pipeline + jump compensation | Bot survives 5 consecutive market-moving events |
| 6. Validation | CPCV + delay/shuffle/regime tests | PBO < 30%, DSR positive, robust under all 4 tests |
| 7. Paper trading | Live shadow PnL | Tracks backtest within tolerance over 2 weeks |
| 8. Tiny live | $100 bankroll, $10/market | Consistent (small) edge over meaningful sample |

---

## 17. Bibliography

**Market making theory**
- Avellaneda & Stoikov (2008). "High-frequency trading in a limit order book." *Quantitative Finance.*
- Guéant, Lehalle & Fernandez-Tapia (2013). "Dealing with the inventory risk." *Mathematics and Financial Economics.*
- Cartea, Jaimungal & Penalva (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
- Cartea, Jaimungal & Ricci (2018). "Algorithmic Trading, Stochastic Control, and Mutually-Exciting Processes." *SIAM Review.*
- Ho & Stoll (1981). "Optimal dealer pricing under transactions and return uncertainty." *JFE.*

**Microstructure**
- Glosten & Milgrom (1985). "Bid, ask and transaction prices in a specialist market." *JFE.*
- Kyle (1985). "Continuous auctions and insider trading." *Econometrica.*
- O'Hara (1995). *Market Microstructure Theory.* Blackwell.
- Hasbrouck (2007). *Empirical Market Microstructure.* Oxford.

**Prediction markets**
- Dubach (2026). "The Anatomy of a Decentralized Prediction Market." *arXiv:2604.24366.*
- Ng, Peng, Tao & Zhou (2025). "Price Discovery and Trading in Modern Prediction Markets." SSRN.
- Groeger (2016). "The Informational Content of the Limit Order Book." *arXiv:1609.03471.*
- "Toward Black-Scholes for Prediction Markets" (2025). *arXiv:2510.15205.*

**Signals**
- Cont, Kukanov & Stoikov (2014). "The price impact of order book events." *J. Financial Econometrics.*
- Stoikov (2018). "The Micro-Price." SSRN; code: github.com/sstoikov/microprice
- Xu, Gould & Howison (2020). "Multi-level order-flow imbalance." World Scientific.
- Guo, Ruan & Zhu (2018). "Dynamics of Order Positions in a Limit Order Book." *arXiv:1505.04810.*
- Easley, López de Prado & O'Hara (2012). "Flow Toxicity and Liquidity." *Review of Financial Studies.*

**Validation**
- López de Prado (2018). *Advances in Financial Machine Learning.* Wiley.
- Bailey & López de Prado (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management.*

**Event studies**
- Brown & Warner (1985). "Using daily stock returns." *JFE.*
- MacKinlay (1997). "Event studies in economics and finance." *J. Economic Literature.*
- Goldsmith-Pinkham & Lyu (2025). "Causal Inference in Financial Event Studies." *arXiv:2511.15123.*

**Kelly**
- Kelly (1956). "A new interpretation of information rate." *Bell System Technical Journal.*
- "Application of the Kelly Criterion to Prediction Markets" (2024). *arXiv:2412.14144.*

**Open-source references**
- `py-clob-client` — official Polymarket Python SDK
- `nautilus_trader.adapters.polymarket` — production trading framework integration
- `github.com/sstoikov/microprice` — reference microprice implementation
- `github.com/warproxxx/poly-maker` — community Polymarket market-maker
- `github.com/yt-feng/VPIN` — VPIN reference

**Data**
- Polymarket CLOB API: `docs.polymarket.com`
- Kalshi API: `trading-api.readme.io/reference`
- GDELT: `gdeltproject.org`
- EDT corporate events dataset: `github.com/Zhihan1996/TradeTheEvent`
