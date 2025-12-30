# analysi/templatetags/extra_filters.py
from django import template

register = template.Library()

@register.filter
def map(sequence, key):
    """Extract list of values from list of dicts"""
    if not sequence:
        return []
    return [item.get(key, 0) for item in sequence]

@register.filter
def get_item(dictionary, key):
    """Safely get value from dict"""
    if isinstance(dictionary, dict):
        return dictionary.get(str(key), "N/A")
    return "N/A"

@register.filter
def split(value, delimiter="_"):
    """Split string by delimiter"""
    if not value:
        return []
    return value.split(delimiter)

# ADD THIS NEW FILTER
@register.filter
def sum_values(values):
    """Sum a list of numbers"""
    try:
        return sum(int(x) for x in values if x is not None)
    except:
        return 0