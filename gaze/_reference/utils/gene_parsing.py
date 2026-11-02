"""
Canonical gene rule parsing for metabolic network GPR (gene-protein-reaction) rules.

All gene rule parsing across the codebase should import from this module.
"""

import re
from typing import List, Tuple


def parse_gene_rule(gene_rule: str) -> Tuple[List[str], str]:
    """
    Parse a gene rule string to extract genes and operation type.

    Args:
        gene_rule: String like "gene1 and gene2" or "gene3 or gene4".
                   May contain nested parentheses.

    Returns:
        (genes, operation): List of gene IDs and operation type
                            ('and', 'or', 'simple', or 'complex').
    """
    if not gene_rule or gene_rule.strip() == '':
        return [], 'simple'

    # Clean up the rule
    cleaned_rule = re.sub(r'[\(\)]', '', gene_rule.strip())

    # Determine operation type
    if ' and ' in cleaned_rule.lower():
        operation = 'and'
        genes = re.split(r' and ', cleaned_rule, flags=re.IGNORECASE)
    elif ' or ' in cleaned_rule.lower():
        operation = 'or'
        genes = re.split(r' or ', cleaned_rule, flags=re.IGNORECASE)
    else:
        operation = 'simple'
        genes = [cleaned_rule]

    # Clean gene names
    genes = [gene.strip() for gene in genes if gene.strip()]

    return genes, operation


def parse_gene_rule_simple(gene_rule: str) -> List[str]:
    """
    Extract all gene IDs from a gene rule string, splitting by both 'and' and 'or'.

    Unlike ``parse_gene_rule`` (which returns the dominant operation type and may
    leave mixed-operator sub-expressions unsplit), this function always splits on
    *both* operators simultaneously, guaranteeing every individual gene ID is
    returned.  This matches the behaviour expected by most callers that only need
    the flat list of gene IDs.

    Args:
        gene_rule: Gene rule string (e.g., "GENE1 and GENE2 or GENE3")

    Returns:
        List of gene IDs extracted from the rule.
    """
    if not gene_rule or (isinstance(gene_rule, str) and gene_rule.strip() == ''):
        return []

    # Remove parentheses
    cleaned_rule = re.sub(r'[\(\)]', '', gene_rule.strip())

    # Split by both 'and' and 'or' operators simultaneously (case-insensitive)
    gene_ids = re.split(r'\s+and\s+|\s+or\s+', cleaned_rule, flags=re.IGNORECASE)

    return [g.strip() for g in gene_ids if g.strip()]
