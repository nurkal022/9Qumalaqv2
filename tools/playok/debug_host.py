"""
Debug opponent: logs in one account, HOSTS a table, invites TARGET, auto-starts,
and plays a RANDOM legal move each turn (using the op-90 legal-move mask).
Used to integration-test bridge.py --accept against a live second account.

Env: H_USER/H_PW (host), TARGET (nick to invite, e.g. alemgamer)
"""
from __future__ import annotations
import os
import random
import time

from playok import PlayokClient

SKIP = {25, 51, 27, 31, 30, 32, 33, 28, 23, 22}


def legal_mask(i):
    for k in range(2, len(i) - 1):
        if i[k] == 5 and isinstance(i[k + 1], int) and 0 <= i[k + 1] <= 511:
            return i[k + 1]
    return None


def main():
    c = PlayokClient(os.environ["H_USER"], os.environ["H_PW"], verbose=False)
    target = os.environ["TARGET"]
    st = {"k": None, "started": False, "handled": False}

    def on(i, s):
        op = i[0] if i else -1
        if op == 84 and len(i) > 1:
            st["k"] = i[1]
        elif op == 29 and s and str(s[-1]).startswith("#"):
            try:
                st["k"] = int(s[-1][1:])
            except ValueError:
                pass
        if (op == 70 and st["k"] and len(i) > 1 and i[1] == st["k"]
                and len(s) >= 3 and s[1] and s[2] and not st["started"]):
            st["started"] = True
            print(f"[H] both seated ({s[1]} vs {s[2]}) -> start", flush=True)
            c.start(st["k"])
        if op == 92:
            st["handled"] = False             # a move happened -> new turn coming
        if op == 90:
            m = legal_mask(i)
            if m and not st["handled"]:
                holes = [h for h in range(9) if m & (1 << h)]
                if holes:
                    h = random.choice(holes)
                    st["handled"] = True
                    print(f"[H] my turn legal={holes} -> play hole {h}", flush=True)
                    c.move(st["k"], h, 5)
        if op not in SKIP:
            print(f"[H] i={i} s={s}", flush=True)

    c.login()
    c.on_frame = on
    c.connect()
    time.sleep(2)
    c.new_table()
    time.sleep(2)
    K = st["k"]
    print(f"[H] hosting PRIVATE table #{K}, inviting {target}", flush=True)
    c.set_table_type(K, closed=True)      # block randoms
    c.sit(K, 0)
    time.sleep(1)
    c.invite(K, target)
    time.sleep(int(os.environ.get("SECS", "150")))
    c.close()
    print("[H] done", flush=True)


if __name__ == "__main__":
    main()
