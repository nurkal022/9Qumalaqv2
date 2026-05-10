import { createHashRouter, RouterProvider, Outlet } from "react-router";
import Header from "./components/layout/Header";
import Lobby from "./routes/Lobby";
import Game from "./routes/Game";
import History from "./routes/History";
import Replay from "./routes/Replay";
import Login from "./routes/Login";
import Register from "./routes/Register";
import Profile from "./routes/Profile";

function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1"><Outlet /></main>
    </div>
  );
}

const router = createHashRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <Lobby /> },
      { path: "/lobby", element: <Lobby /> },
      { path: "/play/:id", element: <Game /> },
      { path: "/history", element: <History /> },
      { path: "/replay/:id", element: <Replay /> },
      { path: "/login", element: <Login /> },
      { path: "/register", element: <Register /> },
      { path: "/profile", element: <Profile /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
