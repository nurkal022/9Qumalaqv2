"""Fast pure-Python Togyz Kumalak board with the REAL rules (verified on 56M PlayOK moves +
770k 9qum states, 2026-09-27): the game ends only when the SIDE TO MOVE has no stones;
then each side sweeps its own board stones into its kazan. >=82 wins, 81-81 draws.
Encoding matches mcts/src/encoding.rs."""
import numpy as np

WIN = 82


class Board:
    __slots__ = ('p', 'k', 't', 's')

    def __init__(self, p=None, k=None, t=None, s=0):
        self.p = p if p is not None else [[9] * 9, [9] * 9]
        self.k = k if k is not None else [0, 0]
        self.t = t if t is not None else [-1, -1]
        self.s = s

    def copy(self):
        return Board([self.p[0][:], self.p[1][:]], self.k[:], self.t[:], self.s)

    @staticmethod
    def parse(pos):
        w, b, k, t, s = pos.strip().split('/')
        return Board([list(map(int, w.split(','))), list(map(int, b.split(',')))],
                     list(map(int, k.split(','))), list(map(int, t.split(','))), int(s))

    def to_str(self):
        return '%s/%s/%d,%d/%d,%d/%d' % (','.join(map(str, self.p[0])), ','.join(map(str, self.p[1])),
                                         self.k[0], self.k[1], self.t[0], self.t[1], self.s)

    def key(self):
        return (tuple(self.p[0]), tuple(self.p[1]), self.k[0], self.k[1], self.t[0], self.t[1], self.s)

    def stones(self):
        return sum(self.p[0]) + sum(self.p[1])

    def legal(self):
        me, opp = self.s, 1 - self.s
        return [i for i in range(9) if self.p[me][i] > 0 and self.t[opp] != i]

    def move(self, pit):
        P, K, T = self.p, self.k, self.t
        player = self.s
        opp = 1 - player
        stones = P[player][pit]
        assert stones > 0 and T[opp] != pit, 'illegal move'
        P[player][pit] = 0
        cp, cs = pit, player
        if stones == 1:
            cp += 1
            if cp > 8:
                cp, cs = 0, opp
            if cs == opp and T[player] == cp:
                K[player] += 1
            elif cs == player and T[opp] == cp:
                K[opp] += 1
            else:
                P[cs][cp] += 1
        else:
            P[cs][cp] += 1
            stones -= 1
            while stones > 0:
                cp += 1
                if cp > 8:
                    cp, cs = 0, 1 - cs
                if cs == opp and T[player] == cp:
                    K[player] += 1
                elif cs == player and T[opp] == cp:
                    K[opp] += 1
                else:
                    P[cs][cp] += 1
                stones -= 1
        is_tuz = (cs == opp and T[player] == cp) or (cs == player and T[opp] == cp)
        if cs == opp and not is_tuz:
            c = P[opp][cp]
            if c == 3 and T[player] == -1 and cp != 8 and T[opp] != cp:
                T[player] = cp
                K[player] += 3
                P[opp][cp] = 0
            elif c > 0 and c % 2 == 0:
                K[player] += c
                P[opp][cp] = 0
        self.s = opp

    def result(self):
        """None, 0 (white wins), 1 (black wins), 2 (draw)."""
        if self.k[0] >= WIN:
            return 0
        if self.k[1] >= WIN:
            return 1
        if not any(self.p[self.s]):
            kw = self.k[0] + sum(self.p[0])
            kb = self.k[1] + sum(self.p[1])
            return 0 if kw > kb else (1 if kb > kw else 2)
        return None

    def final_margin(self):
        """Swept stone difference white-black at a terminal position."""
        return (self.k[0] + sum(self.p[0])) - (self.k[1] + sum(self.p[1]))

    def encode(self):
        me, opp = self.s, 1 - self.s
        x = np.zeros((7, 9), dtype=np.float32)
        x[0] = np.asarray(self.p[me], dtype=np.float32) / 50.0
        x[1] = np.asarray(self.p[opp], dtype=np.float32) / 50.0
        x[2] = self.k[me] / 82.0
        x[3] = self.k[opp] / 82.0
        if self.t[me] >= 0:
            x[4, self.t[me]] = 1.0
        if self.t[opp] >= 0:
            x[5, self.t[opp]] = 1.0
        x[6] = 1.0 if self.s == 0 else 0.0
        return x
