import { render, screen } from "@testing-library/react";
import App from "./App";
import "./i18n";

test("renders Lobby on /", () => {
  render(<App />);
  // Lobby's hero is the Quick Start button (KK locale)
  expect(screen.getByText("Жылдам бастау")).toBeInTheDocument();
});

test("renders header app title", () => {
  render(<App />);
  expect(screen.getByText("Тоғызқұмалақ")).toBeInTheDocument();
});
