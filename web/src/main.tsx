import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";

import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ToastProvider } from "./components/ui";
import { AppProvider } from "./hooks/useApp";
import VerifyPage from "./pages/VerifyPage";

const BatchPage = lazy(() => import("./pages/BatchPage"));
const EvaluationPage = lazy(() => import("./pages/EvaluationPage"));
const AboutPage = lazy(() => import("./pages/AboutPage"));

function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-4 py-24 text-center">
      <p className="text-sm font-medium text-accent-ink">404</p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Page not found</h1>
      <Link to="/" className="mt-6 inline-block text-sm font-medium text-accent-ink hover:underline">Verify a claim →</Link>
    </div>
  );
}

const Fallback = () => <div className="mx-auto max-w-7xl px-6 py-16"><div className="shimmer h-8 w-64 rounded-lg" /></div>;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <AppProvider>
        <ToastProvider>
          <Layout>
            <Suspense fallback={<Fallback />}>
              <Routes>
                <Route path="/" element={<VerifyPage />} />
                <Route path="/batch" element={<BatchPage />} />
                <Route path="/evaluation" element={<EvaluationPage />} />
                <Route path="/how-it-works" element={<AboutPage />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Suspense>
          </Layout>
        </ToastProvider>
      </AppProvider>
    </BrowserRouter>
  </StrictMode>,
);
