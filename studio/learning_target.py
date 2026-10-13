"""Resolve optional target descriptions without writing defaults into user state."""

DEFAULT_TARGETS = {
    'character': (
        'Learn the likeness of the person or character represented by the dataset. '
        'By default, clothing or visible state of undress, pose, expression, hairstyle, '
        'accessories, background, lighting and framing remain changeable. '
        'Use the selected Keep/Change rules for any overrides. '
        'Do not infer identity or unseen attributes; flag ambiguous subjects for review.'
    ),
    'object': (
        'Learn the appearance of the main object or product represented by the dataset. '
        'Use the selected Keep/Change rules to separate its defining features from context. '
        'If several objects are plausible targets, request a distinguishing hint or manual '
        'selection rather than guessing. Do not infer hidden features.'
    ),
    'style': (
        'Learn the visual treatment represented by the dataset, using the selected style '
        'family and Keep/Change rules. Depicted subjects and scene content remain changeable '
        'by default. Do not infer an artist or unseen production techniques.'
    ),
    'concept': (
        'Learn the visible concept represented by the dataset, using the selected concept '
        'family and Keep/Change rules to define what repeats. Participants and unrelated '
        'context remain changeable by default. Flag unclear target evidence for review; '
        'do not invent actions, intentions or hidden relationships.'
    ),
}


def effective_description(project):
    """Explicit text wins; whitespace-only input uses the current type's default."""
    return project.get('description', '').strip() or DEFAULT_TARGETS[project['kind']]
