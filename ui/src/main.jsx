import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import StromSageV3 from "./StromSagev3.jsx";
import StromSageUI from "./StromSageUI.jsx";

function App() {
  const [mode, setMode] = useState("chat");

  if (mode === "advisory") {
    return <StromSageV3 onClose={() => setMode("chat")} />;
  }

  return <StromSageUI onOpenAdvisory={() => setMode("advisory")} />;
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
