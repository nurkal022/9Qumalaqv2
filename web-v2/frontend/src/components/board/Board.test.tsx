import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import Board from "./Board";
import { parsePos } from "../../domain/board";

const START = "9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0";

test("renders 18 holes for the start position", () => {
  const layout = parsePos(START);
  render(<Board layout={layout} side={0} sideToMove={0} onMove={() => {}} />);
  expect(screen.getByTestId("my-row").children).toHaveLength(9);
  expect(screen.getByTestId("opp-row").children).toHaveLength(9);
});

test("hole is disabled when not viewer's turn", () => {
  const layout = parsePos(START);
  render(<Board layout={layout} side={0} sideToMove={1} onMove={() => {}} />);
  const myHoles = screen.getByTestId("my-row").querySelectorAll("button");
  myHoles.forEach((b) => expect(b).toBeDisabled());
});

test("empty pit is disabled even on your turn", () => {
  // Custom position: white's pit 0 is empty
  const pos = "0,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0";
  const layout = parsePos(pos);
  render(<Board layout={layout} side={0} sideToMove={0} onMove={() => {}} />);
  const myHoles = screen.getByTestId("my-row").querySelectorAll("button");
  expect(myHoles[0]).toBeDisabled();   // empty
  expect(myHoles[1]).not.toBeDisabled(); // 9 stones
});

describe("parsePos", () => {
  it("parses start position correctly", () => {
    const layout = parsePos("9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0");
    expect(layout.bottomRow).toHaveLength(9);
    expect(layout.bottomRow.every((h) => h.pebbles === 9)).toBe(true);
    expect(layout.whiteKazan).toBe(0);
    expect(layout.blackKazan).toBe(0);
    expect(layout.sideToMove).toBe(0);
    expect(layout.whiteTuz).toBe(-1);
    expect(layout.blackTuz).toBe(-1);
  });

  it("parses kazan values correctly", () => {
    const layout = parsePos("5,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/10,5/-1,-1/1");
    expect(layout.whiteKazan).toBe(10);
    expect(layout.blackKazan).toBe(5);
    expect(layout.sideToMove).toBe(1);
  });

  it("parses tuzdyk indices correctly", () => {
    // white tuzdyk on black's pit 3, black tuzdyk on white's pit 5
    const layout = parsePos("9,9,9,9,9,0,9,9,9/9,9,9,0,9,9,9,9,9/15,20/3,5/0");
    expect(layout.whiteTuz).toBe(3);
    expect(layout.blackTuz).toBe(5);
    // white's tuz is on black's row (topRow), black's tuz is on white's row (bottomRow)
    expect(layout.topRow[3].isTuz).toBe(true);
    expect(layout.bottomRow[5].isTuz).toBe(true);
    // others should NOT be tuz
    expect(layout.topRow[0].isTuz).toBe(false);
    expect(layout.bottomRow[0].isTuz).toBe(false);
  });

  it("throws on invalid format", () => {
    expect(() => parsePos("bad/format")).toThrow("invalid pos format");
  });
});
