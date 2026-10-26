import { isValidElement, type ChangeEvent, type FormEvent, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatEmailGate from "./ChatEmailGate";

// Keep hook state between explicit renders while exercising the real handlers.
const hooks = vi.hoisted(() => ({ values: [] as unknown[], cursor: 0 }));
vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useState: (initial: unknown) => {
    const slot = hooks.cursor++;
    if (!(slot in hooks.values)) hooks.values[slot] = initial;
    return [hooks.values[slot], (next: unknown) => { hooks.values[slot] = next; }];
  },
}));

type GateProps = Parameters<typeof ChatEmailGate>[0];
type Element = { type: unknown; props: Record<string, unknown> };

function render(props: GateProps): ReactNode {
  hooks.cursor = 0;
  return ChatEmailGate(props);
}

function elements(node: ReactNode): Element[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!isValidElement<Record<string, unknown>>(node)) return [];
  return [{ type: node.type, props: node.props }, ...elements(node.props.children as ReactNode)];
}

function element(tree: ReactNode, type: string): Record<string, unknown> {
  const found = elements(tree).find((item) => item.type === type);
  if (!found) throw new Error(`Missing element: ${type}`);
  return found.props;
}

function enterEmail(props: GateProps, value: string): ReactNode {
  const input = element(render(props), "input");
  (input.onChange as (event: ChangeEvent<HTMLInputElement>) => void)(
    { target: { value } } as ChangeEvent<HTMLInputElement>,
  );
  return render(props);
}

async function submit(tree: ReactNode): Promise<void> {
  const preventDefault = vi.fn();
  await (element(tree, "form").onSubmit as (event: FormEvent) => Promise<void>)(
    { preventDefault } as unknown as FormEvent,
  );
  expect(preventDefault).toHaveBeenCalledOnce();
}

beforeEach(() => {
  hooks.values = [];
  hooks.cursor = 0;
});

describe("email before chat interactions", () => {
  it.each(["", "   ", "not-an-email", "buyer@example"])("blocks missing or invalid email: %j", async (email) => {
    const onContinue = vi.fn();
    const props = { onContinue };
    const tree = enterEmail(props, email);
    expect(element(tree, "button").disabled).toBe(true);
    await submit(tree);
    expect(onContinue).not.toHaveBeenCalled();
    const next = render(props);
    expect(element(next, "input")["aria-invalid"]).toBe(true);
    expect(elements(next).find((item) => item.props.role === "alert")?.props.children).toBe("Please enter a valid email address.");
  });

  it("submits a trimmed address and locks the form until saving completes", async () => {
    let finish!: () => void;
    const onContinue = vi.fn(() => new Promise<void>((resolve) => { finish = resolve; }));
    const props = { onContinue };
    const tree = enterEmail(props, "  buyer@example.test  ");
    expect(element(tree, "button").disabled).toBe(false);
    const pending = submit(tree);
    expect(onContinue).toHaveBeenCalledExactlyOnceWith("buyer@example.test");
    const saving = render(props);
    expect(element(saving, "input").disabled).toBe(true);
    expect(element(saving, "button").disabled).toBe(true);
    expect(element(saving, "button").children).toBe("Saving…");
    await submit(saving);
    expect(onContinue).toHaveBeenCalledOnce();
    finish();
    await pending;
    const saved = render(props);
    expect(element(saved, "button").disabled).toBe(false);
    expect(element(saved, "button").children).toBe("Start chat");
    expect(elements(saved).some((item) => item.props.role === "alert")).toBe(false);
  });

  it("displays a save failure and lets the customer retry", async () => {
    const onContinue = vi.fn()
      .mockRejectedValueOnce(new Error("Could not save email. Please try again."))
      .mockResolvedValueOnce(undefined);
    const props = { onContinue };
    await submit(enterEmail(props, "buyer@example.test"));
    const failed = render(props);
    expect(elements(failed).find((item) => item.props.role === "alert")?.props.children).toBe("Could not save email. Please try again.");
    expect(element(failed, "input").value).toBe("buyer@example.test");
    expect(element(failed, "input").disabled).toBeFalsy();
    expect(element(failed, "button").disabled).toBe(false);
    await submit(failed);
    expect(onContinue).toHaveBeenCalledTimes(2);
    expect(elements(render(props)).some((item) => item.props.role === "alert")).toBe(false);
  });

  it("keeps a disabled gate closed even when its email is valid", async () => {
    const onContinue = vi.fn();
    const props = { onContinue, disabled: true };
    const tree = enterEmail(props, "buyer@example.test");
    expect(element(tree, "input").disabled).toBe(true);
    expect(element(tree, "button").disabled).toBe(true);
    await submit(tree);
    expect(onContinue).not.toHaveBeenCalled();
  });
});
