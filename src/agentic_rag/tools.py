"""Pydantic-validated tools the executor agent can call."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from pydantic import BaseModel, Field, ValidationError, field_validator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


class CalculatorInput(BaseModel):
    expression: str = Field(..., max_length=200, description="Arithmetic expression, e.g. '12 * (3 + 4)'")

    @field_validator("expression")
    @classmethod
    def only_math(cls, v: str) -> str:
        if not set(v) <= set("0123456789.+-*/()% "):
            raise ValueError("expression may only contain numbers and + - * / % ( )")
        return v


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_safe_eval(node.right)) > 10:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(args: CalculatorInput) -> str:
    result = _safe_eval(ast.parse(args.expression, mode="eval"))
    return f"{args.expression} = {round(result, 6):g}"


class Tool(BaseModel):
    name: str
    description: str
    schema_: type[BaseModel]
    fn: Callable[[BaseModel], str]

    model_config = {"arbitrary_types_allowed": True}


TOOLS: dict[str, Tool] = {
    "calculator": Tool(
        name="calculator",
        description="Evaluate an arithmetic expression exactly.",
        schema_=CalculatorInput,
        fn=calculator,
    ),
}


class ToolError(Exception):
    pass


def run_tool(name: str, raw_input: str) -> str:
    """Validate input against the tool's schema, then execute it."""
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"unknown tool '{name}'")
    field_name = next(iter(tool.schema_.model_fields))
    try:
        args = tool.schema_(**{field_name: raw_input})
        return tool.fn(args)
    except (ValidationError, ValueError, ZeroDivisionError, SyntaxError) as e:
        raise ToolError(f"{name} failed: {e}") from e
