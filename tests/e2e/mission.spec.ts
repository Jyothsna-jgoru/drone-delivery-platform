import { expect, test } from "@playwright/test";

test("create, validate, simulate, inspect and open report", async ({ page, request }) => {
  await page.goto("http://localhost:5173");
  await expect(page.getByText("Mission Center", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Validate mission" }).click();
  await expect(page.getByText("All deterministic checks passed")).toBeVisible();
  await page.getByRole("button", { name: "Submit" }).click();
  await page.getByRole("button", { name: "Launch simulation" }).click();
  await page.getByRole("button", { name: "Live Fleet" }).click();
  await expect(page.getByText("Live simulator telemetry")).toBeVisible();
  await page.getByRole("button", { name: "Decision Inspector" }).click();
  await expect(page.getByText(/RECORDED DECISION/)).toBeVisible();
  await page.getByRole("button", { name: "Reports" }).click();
  await expect(page.getByRole("heading", { name: "Mission report" })).toBeVisible();
});

