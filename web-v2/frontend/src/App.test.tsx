import { render, screen } from "@testing-library/react";
import "./i18n";
import App from "./App";

test("renders title", () => {
  render(<App />);
  expect(screen.getByText("Тоғызқұмалақ")).toBeInTheDocument();
});
