"""Policy configuration loader from brand YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import yaml

from src.policies.schemas import BrandPolicyConfig

CONFIG_DIR = Path("config")


def load_brand_policy(brand_id: str, config_dir: Path = CONFIG_DIR) -> BrandPolicyConfig:
    """Loads and parses the policy section of a brand's YAML file.

    Args:
        brand_id: Unique brand identifier (e.g. 'lumora', 'urbankicks').
        config_dir: Directory containing brand configuration YAMLs.

    Returns:
        Validated BrandPolicyConfig instance.
    """
    config_file = config_dir / f"{brand_id.lower()}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found for brand: {brand_id} at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    policies = data.get("policies", {})
    non_returnable = policies.get("non_returnable_categories", [])
    
    is_skincare = "skincare" in data.get("category", "").lower()
    is_apparel = "streetwear" in data.get("category", "").lower() or "sneakers" in data.get("category", "").lower()

    return BrandPolicyConfig(
        brand_id=data.get("brand_id", brand_id),
        allowed_return_days=policies.get("return_window_days", 14),
        allowed_exchange_days=policies.get("exchange_window_days", 14),
        non_returnable_categories=non_returnable,
        require_unopened_for_skincare=is_skincare,
        require_unworn_for_apparel=is_apparel,
        allow_defective_exceptions=True,
        allow_wrong_item_exceptions=True,
    )
