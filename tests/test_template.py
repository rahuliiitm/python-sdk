from launchpromptly.template import extract_variables, interpolate


class TestExtractVariables:
    def test_finds_all_variables(self):
        result = extract_variables("Hello {{name}}, your role is {{role}}. Email: {{email}}")
        assert result == ["name", "role", "email"]

    def test_empty_for_no_variables(self):
        assert extract_variables("No variables here") == []

    def test_deduplicates(self):
        assert extract_variables("{{x}} and {{x}} again") == ["x"]

    def test_single_variable(self):
        assert extract_variables("Hi {{name}}") == ["name"]


class TestInterpolate:
    def test_replaces_all_variables(self):
        result = interpolate("Hi {{name}}, welcome to {{place}}!", {"name": "World", "place": "Earth"})
        assert result == "Hi World, welcome to Earth!"

    def test_leaves_unmatched_as_is(self):
        result = interpolate("{{a}} + {{b}} = {{c}}", {"a": "1", "b": "2"})
        assert result == "1 + 2 = {{c}}"

    def test_handles_special_chars(self):
        result = interpolate("Price: {{amount}}", {"amount": "$100.00 (USD)"})
        assert result == "Price: $100.00 (USD)"

    def test_empty_variables(self):
        result = interpolate("Hello {{name}}", {})
        assert result == "Hello {{name}}"
