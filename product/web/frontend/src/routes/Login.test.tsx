import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import Login from "./Login";
import "../i18n";

test("login form renders fields", () => {
  render(
    <MemoryRouter>
      <Login />
    </MemoryRouter>,
  );
  // Two inputs — username and password
  expect(document.querySelectorAll("input").length).toBe(2);
  expect(screen.getByRole("button")).toBeInTheDocument();
});
