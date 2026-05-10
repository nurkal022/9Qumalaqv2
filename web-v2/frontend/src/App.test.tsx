import { render, screen } from "@testing-library/react";
import App from "./App";
import "./i18n";

test("renders Lobby on /", () => {
  render(<App />);
  // App.tsx uses HashRouter; default route is Lobby — check form submit button (KK locale)
  expect(screen.getByRole("button", { name: "Бастау" })).toBeInTheDocument();
});

test("renders header app title", () => {
  render(<App />);
  expect(screen.getByText("Тоғызқұмалақ")).toBeInTheDocument();
});
