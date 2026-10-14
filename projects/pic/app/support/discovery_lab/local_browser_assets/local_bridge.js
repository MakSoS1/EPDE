// Trusted Streamlit component JavaScript. Data is passed as a plain object;
// never interpolate user data into this JS or markup. No external scripts.
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector(".epde-local");
  const start = root.querySelector(".epde-start");
  const check = root.querySelector(".epde-check");
  const status = root.querySelector(".epde-status");
  let controller = null;
  let finished = false;
  const uri = "http://127.0.0.1:8765";
  const headers = {
    "Content-Type": "application/json",
    "X-EPDE-Token": String(data?.token || ""),
  };
  function show(message) { status.textContent = String(message); }
  function busy(b) { start.disabled = b; check.disabled = b; }

  check.addEventListener("click", async () => {
    show("Connecting to local worker…");
    busy(true);
    controller = new AbortController();
    try {
      const res = await fetch(uri + "/health", {
        method: "GET", headers, mode: "cors", cache: "no-store",
        signal: controller.signal,
      });
      if (!res.ok) throw new Error("Worker rejected access code or browser origin.");
      const body = await res.json();
      if (!body.ok || body.protocol !== "discovery-local/1.0") {
        throw new Error("Unexpected local worker version.");
      }
      show("Connected. This computer can run EPDE and PySINDy.");
    } catch (err) {
      show("Cannot connect: " + err.message +
           " Start local_worker_service.py, check its access code, and allow browser local-network access.");
    } finally { busy(false); }
  });

  start.addEventListener("click", async () => {
    if (finished) return;
    busy(true);
    show("Running on your computer. Keep this tab and your local worker open…");
    controller = new AbortController();
    try {
      const res = await fetch(uri + "/run", {
        method: "POST", headers, mode: "cors",
        body: JSON.stringify(data?.job || {}), cache: "no-store",
        signal: controller.signal,
      });
      const body = await res.json();
      if (!res.ok || !body.record) {
        throw new Error(String(body.error || ("Local worker returned HTTP " + res.status)));
      }
      const record = body.record;
      if (record.execution_location !== "user_computer" ||
          !["epde", "pysindy"].includes(record.method)) {
        throw new Error("Invalid scientific result returned by the worker.");
      }
      const size = JSON.stringify(record).length;
      if (size > 1800000) throw new Error("Result too large to upload to the site.");
      finished = true;
      show("Finished. Returning results to the site's Results page…");
      setTriggerValue("result", record);
    } catch (err) {
      show("Local run failed: " + String(err.message));
      setTriggerValue("failed", String(err.message).slice(0, 220));
    } finally {
      if (!finished) busy(false);
    }
  });
  return () => {
    if (controller) controller.abort();
  };
}
