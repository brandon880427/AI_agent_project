# -*- coding: utf-8 -*-
"""Texas Hold'em preflop equity / simple EV helpers.

Designed to be dependency-free and reasonably fast for small Monte Carlo runs.
This is *not* a full solver; it estimates equity vs random opponent hands.

Public API:
- parse_card("As") -> (rank_int, suit_int)
- estimate_equity(hole_cards, num_opponents=1, iters=2000, seed=None)
- advise_action(equity, pot_before_call=None, to_call=None)

Ranks: 2..14 (A=14)
Suits: 0..3 (s,h,d,c)
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Sequence, Tuple


_RANK_TO_INT = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

_SUIT_TO_INT = {
    "S": 0,
    "♠": 0,
    "SPADES": 0,
    "H": 1,
    "♥": 1,
    "HEARTS": 1,
    "D": 2,
    "♦": 2,
    "DIAMONDS": 2,
    "C": 3,
    "♣": 3,
    "CLUBS": 3,
}

_INT_TO_SUIT = {0: "S", 1: "H", 2: "D", 3: "C"}
_INT_TO_RANK = {v: k for k, v in _RANK_TO_INT.items() if k not in {"T"}}
_INT_TO_RANK[10] = "T"


def parse_card(s: str) -> Tuple[int, int]:
    """Parse a card string like 'As', 'TD', '10h', 'A♠', 'K♣'."""
    if not s:
        raise ValueError("empty card")
    t = str(s).strip()
    if not t:
        raise ValueError("empty card")

    # normalize unicode suit at end (A♠)
    suit_ch = t[-1]
    rank_part = t[:-1]

    if suit_ch in _SUIT_TO_INT:
        suit = _SUIT_TO_INT[suit_ch]
    else:
        suit = _SUIT_TO_INT.get(suit_ch.upper())
    if suit is None:
        # sometimes input like 'AS' already
        suit = _SUIT_TO_INT.get(suit_ch.upper())
    if suit is None:
        raise ValueError(f"invalid suit in card: {s}")

    rp = rank_part.strip().upper()
    if rp == "1":
        rp = "A"
    if rp in _RANK_TO_INT:
        rank = _RANK_TO_INT[rp]
    else:
        # allow '10' already handled, but fallback
        raise ValueError(f"invalid rank in card: {s}")

    return rank, suit


def format_card(card: Tuple[int, int]) -> str:
    r, su = card
    return f"{_INT_TO_RANK.get(int(r), str(r))}{_INT_TO_SUIT.get(int(su), '?')}"


def _deck52() -> List[Tuple[int, int]]:
    d: List[Tuple[int, int]] = []
    for su in range(4):
        for r in range(2, 15):
            d.append((r, su))
    return d


def _straight_high(ranks: Iterable[int]) -> int:
    """Return high card of the best straight; 0 if none.

    ranks are unique ranks 2..14.
    """
    rs = set(int(r) for r in ranks)
    if 14 in rs:
        rs.add(1)  # wheel
    for high in range(14, 4, -1):
        if all((high - i) in rs for i in range(5)):
            return high
    return 0


def _hand_rank_7(cards: Sequence[Tuple[int, int]]) -> Tuple[int, Tuple[int, ...]]:
    """Evaluate a 7-card hand.

    Returns (category, tiebreakers) where larger is better.

    Categories:
      8: straight flush
      7: four of a kind
      6: full house
      5: flush
      4: straight
      3: three of a kind
      2: two pair
      1: one pair
      0: high card
    """
    ranks = [int(r) for r, _ in cards]
    suits = [int(s) for _, s in cards]

    # counts by rank
    count_by_rank: dict[int, int] = {}
    for r in ranks:
        count_by_rank[r] = count_by_rank.get(r, 0) + 1

    # flush detection
    suit_counts = [0, 0, 0, 0]
    for s in suits:
        suit_counts[s] += 1
    flush_suit = -1
    for s, c in enumerate(suit_counts):
        if c >= 5:
            flush_suit = s
            break

    # straight / straight flush
    uniq_ranks = set(ranks)
    straight_hi = _straight_high(uniq_ranks)

    if flush_suit != -1:
        flush_ranks = [r for (r, s) in cards if s == flush_suit]
        flush_uniq = set(flush_ranks)
        sf_hi = _straight_high(flush_uniq)
        if sf_hi:
            return 8, (sf_hi,)

    # group ranks by count
    groups = sorted(((cnt, r) for r, cnt in count_by_rank.items()), reverse=True)
    # groups sorted by count desc then rank desc

    # four of a kind
    if groups and groups[0][0] == 4:
        quad_rank = groups[0][1]
        kicker = max(r for r in uniq_ranks if r != quad_rank)
        return 7, (quad_rank, kicker)

    # full house
    if groups and groups[0][0] == 3:
        trip_rank = groups[0][1]
        pair_rank = 0
        for cnt, r in groups[1:]:
            if cnt >= 2:
                pair_rank = r
                break
        if pair_rank:
            return 6, (trip_rank, pair_rank)

    # flush
    if flush_suit != -1:
        flush_ranks_sorted = sorted((r for (r, s) in cards if s == flush_suit), reverse=True)
        top5 = tuple(flush_ranks_sorted[:5])
        return 5, top5

    # straight
    if straight_hi:
        return 4, (straight_hi,)

    # three of a kind
    if groups and groups[0][0] == 3:
        trip_rank = groups[0][1]
        kickers = sorted((r for r in uniq_ranks if r != trip_rank), reverse=True)[:2]
        return 3, (trip_rank, *kickers)

    # two pair
    if len(groups) >= 2 and groups[0][0] == 2 and groups[1][0] == 2:
        pair1 = groups[0][1]
        pair2 = groups[1][1]
        kicker = max(r for r in uniq_ranks if r not in (pair1, pair2))
        hi, lo = (pair1, pair2) if pair1 >= pair2 else (pair2, pair1)
        return 2, (hi, lo, kicker)

    # one pair
    if groups and groups[0][0] == 2:
        pair = groups[0][1]
        kickers = sorted((r for r in uniq_ranks if r != pair), reverse=True)[:3]
        return 1, (pair, *kickers)

    # high card
    top5 = tuple(sorted(uniq_ranks, reverse=True)[:5])
    return 0, top5


def estimate_equity(
    hole_cards: Sequence[Tuple[int, int]],
    *,
    num_opponents: int = 1,
    iters: int = 2000,
    seed: Optional[int] = None,
) -> float:
    """Monte Carlo equity vs random hands.

    Equity is computed as average fraction of the pot won (ties split).
    """
    if len(hole_cards) != 2:
        raise ValueError("hole_cards must have length 2")
    if num_opponents < 1 or num_opponents > 8:
        raise ValueError("num_opponents must be 1..8")
    if iters < 100:
        raise ValueError("iters must be >= 100")

    hc0, hc1 = hole_cards[0], hole_cards[1]
    if hc0 == hc1:
        raise ValueError("duplicate hole cards")

    rng = random.Random(seed)

    deck = _deck52()
    dead = {hc0, hc1}
    deck = [c for c in deck if c not in dead]

    hero_wins_frac = 0.0

    need = (2 * num_opponents) + 5
    for _ in range(iters):
        sample = rng.sample(deck, need)
        board = sample[:5]
        opp_cards = sample[5:]

        hero7 = [hc0, hc1, *board]
        hero_rank = _hand_rank_7(hero7)

        best_rank = hero_rank
        winners = [0]  # 0=hero, others=opponents

        # evaluate opponents
        for oi in range(num_opponents):
            o0 = opp_cards[2 * oi]
            o1 = opp_cards[2 * oi + 1]
            o7 = [o0, o1, *board]
            orank = _hand_rank_7(o7)
            if orank > best_rank:
                best_rank = orank
                winners = [oi + 1]
            elif orank == best_rank:
                winners.append(oi + 1)

        if best_rank == hero_rank:
            # hero is among winners
            if 0 in winners:
                hero_wins_frac += 1.0 / float(len(winners))

    return hero_wins_frac / float(iters)


@dataclass(frozen=True)
class Advice:
    action: str
    equity: float
    pot_odds: Optional[float]
    ev_call: Optional[float]
    reason: str


def advise_action(
    equity: float,
    *,
    pot_before_call: Optional[float] = None,
    to_call: Optional[float] = None,
) -> Advice:
    """Very simple preflop decision helper.

    - If to_call is provided, compare equity vs pot odds.
    - If no to_call is provided, suggest 'bet' for strong hands and 'fold' for weak.

    This is intentionally simple and should be treated as a baseline.
    """
    e = float(max(0.0, min(1.0, equity)))

    if to_call is None or pot_before_call is None:
        # No pot odds context: just provide a generic action.
        if e >= 0.62:
            return Advice("bet", e, None, None, "牌力很强（对随机手牌胜率高），可主动下注争取价值")
        if e >= 0.52:
            return Advice("call", e, None, None, "牌力中上，可继续游戏（视位置与下注大小调整）")
        return Advice("fold", e, None, None, "牌力偏弱，建议弃牌以避免长期负期望")

    P = float(pot_before_call)
    C = float(to_call)
    if C < 0:
        C = 0.0
    if P < 0:
        P = 0.0

    pot_odds = C / (P + C) if (P + C) > 0 else 1.0
    ev_call = e * (P + C) - (1.0 - e) * C

    # small margin to avoid flip-flopping
    margin = 0.02
    if e + margin < pot_odds:
        return Advice("fold", e, pot_odds, ev_call, "胜率低于赔率需求，跟注为负期望")

    # positive EV
    if C == 0:
        if e >= 0.6:
            return Advice("bet", e, pot_odds, ev_call, "无需跟注成本且胜率高，适合主动下注")
        return Advice("check", e, pot_odds, ev_call, "无需跟注成本，偏向check/看免费牌")

    if e >= 0.7:
        return Advice("raise", e, pot_odds, ev_call, "胜率很高，可考虑加注获取价值（仍需考虑位置与对手范围）")

    return Advice("call", e, pot_odds, ev_call, "胜率满足赔率需求，跟注为正期望")
