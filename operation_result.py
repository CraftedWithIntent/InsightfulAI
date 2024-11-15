# operation_result.py
from typing import Generic, TypeVar, Optional, Callable
from opentelemetry import trace
from opentelemetry.trace import Span

T = TypeVar("T")
U = TypeVar("U")

class OperationResult(Generic[T]):
    def __init__(self, is_success: bool, result: Optional[T] = None, error: Optional[Exception] = None, span: Optional[Span] = None):
        self.is_success = is_success
        self.result = result
        self.error = error
        self.span = span or trace.get_current_span()  # Use current active span if none provided

    @classmethod
    def success(cls, result: T, span: Optional[Span] = None) -> "OperationResult[T]":
        return cls(is_success=True, result=result, span=span or trace.get_current_span())

    @classmethod
    def failure(cls, error: Exception, span: Optional[Span] = None) -> "OperationResult[T]":
        return cls(is_success=False, error=error, span=span or trace.get_current_span())

    def bind(self, func: Callable[[T], "OperationResult[U]"]) -> "OperationResult[U]":
        """Chains another operation if this one is successful; otherwise, propagates the failure."""
        if self.is_success:
            try:
                # Propagate the span to the next operation in the chain
                return func(self.result).with_span(self.span)
            except Exception as e:
                return OperationResult.failure(e, self.span)
        else:
            return OperationResult.failure(self.error, self.span)

    def with_span(self, span: Optional[Span]) -> "OperationResult[T]":
        """Sets the span for this result, returning a new OperationResult with the updated span."""
        return OperationResult(self.is_success, result=self.result, error=self.error, span=span)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.is_success:
            return f"OperationResult(success, result={self.result}, span_id={self._span_id()})"
        else:
            return f"OperationResult(failure, error={self.error}, span_id={self._span_id()})"

    def _span_id(self) -> str:
        """Helper to extract a unique identifier from the span if available."""
        if self.span:
            try:
                return self.span.get_span_context().span_id
            except AttributeError:
                return "Invalid Span"
        return "None"

    def __format__(self, format_spec: str) -> str:
        """Custom formatting to handle numeric and string representations."""
        if self.is_success and isinstance(self.result, (int, float)):
            return format(self.result, format_spec)
        else:
            return format(str(self), format_spec)