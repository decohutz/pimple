import { createBrowserRouter } from "react-router-dom";
import AppShell from "./layout/AppShell";
import Dashboard from "./pages/Dashboard";
import Dataset from "./pages/Dataset";
import Inference from "./pages/Inference";
import History from "./pages/History";
import About from "./pages/About";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: "dataset", element: <Dataset /> },
      { path: "inference", element: <Inference /> },
      { path: "history", element: <History /> },
      { path: "about", element: <About /> }
    ]
  }
]);
