
from pydantic import BaseModel, Field, model_validator


class ProductFilters(BaseModel):
    color: str | None = None
    category: str | None = None
    brand: str | None = None

    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_price_range(self):
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError(
                "min_price cannot exceed max_price"
            )
        return self


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="Text describing the product to find",
    )

    filters: ProductFilters = Field(
        default_factory=ProductFilters
    )

    top_k: int = Field(default=10, ge=1, le=50)
