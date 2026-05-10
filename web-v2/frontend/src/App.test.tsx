import { render, screen } from "@testing-library/react";
import App from "./App";
import "./i18n";

test("renders Lobby on /", () => {
  render(<App />);
  // App.tsx uses HashRouter; default route is Lobby
  expect(screen.getByText(/Lobby/i)).toBeInTheDocument();
});

test("renders header app title", () => {
  render(<App />);
  expect(screen.getByText("Тоғызқұмалақ")).toBeInTheDocument();
});
