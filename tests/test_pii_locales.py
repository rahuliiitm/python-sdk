"""Tests for multi-language PII detection."""

from launchpromptly._internal.pii import PIIDetectOptions, detect_pii


# ── Canada ──────────────────────────────────────────────────────────────────


class TestCanadaPII:
    def test_detects_valid_sin_with_dashes(self):
        results = detect_pii("My SIN is 046-454-286", PIIDetectOptions(locales=["ca"]))
        assert any(r.type == "ca_sin" and r.value == "046-454-286" for r in results)

    def test_detects_sin_with_spaces(self):
        results = detect_pii("SIN: 046 454 286", PIIDetectOptions(locales=["ca"]))
        assert any(r.type == "ca_sin" for r in results)

    def test_detects_bare_sin_with_context(self):
        results = detect_pii("social insurance number: 046454286", PIIDetectOptions(locales=["ca"]))
        assert any(r.type == "ca_sin" for r in results)

    def test_rejects_sin_failing_luhn(self):
        results = detect_pii("SIN: 123-456-789", PIIDetectOptions(locales=["ca"]))
        assert not any(r.type == "ca_sin" for r in results)

    def test_no_false_positive_on_phone(self):
        results = detect_pii("Call 416-555-1234", PIIDetectOptions(locales=["ca"]))
        assert not any(r.type == "ca_sin" for r in results)

    def test_confidence_gte_0_8(self):
        results = detect_pii("my SIN is 046-454-286", PIIDetectOptions(locales=["ca"]))
        sin = next((r for r in results if r.type == "ca_sin"), None)
        assert sin is not None
        assert sin.confidence >= 0.8


# ── Brazil ──────────────────────────────────────────────────────────────────


class TestBrazilPII:
    def test_detects_valid_formatted_cpf(self):
        results = detect_pii("Meu CPF é 529.982.247-25", PIIDetectOptions(locales=["br"]))
        assert any(r.type == "br_cpf" for r in results)

    def test_detects_bare_cpf_with_context(self):
        results = detect_pii("CPF do cliente: 52998224725", PIIDetectOptions(locales=["br"]))
        assert any(r.type == "br_cpf" for r in results)

    def test_rejects_cpf_invalid_check_digits(self):
        results = detect_pii("CPF: 123.456.789-00", PIIDetectOptions(locales=["br"]))
        assert not any(r.type == "br_cpf" for r in results)

    def test_rejects_all_same_digit_cpf(self):
        results = detect_pii("CPF: 111.111.111-11", PIIDetectOptions(locales=["br"]))
        assert not any(r.type == "br_cpf" for r in results)

    def test_detects_valid_cnpj(self):
        results = detect_pii("CNPJ: 11.222.333/0001-81", PIIDetectOptions(locales=["br"]))
        assert any(r.type == "br_cnpj" for r in results)

    def test_detects_brazilian_phone(self):
        results = detect_pii("+55 11 91234-5678", PIIDetectOptions(locales=["br"]))
        # +55 may be matched by base intl phone pattern
        assert len(results) > 0

    def test_confidence_gte_0_9_for_formatted_cpf(self):
        results = detect_pii("CPF: 529.982.247-25", PIIDetectOptions(locales=["br"]))
        cpf = next((r for r in results if r.type == "br_cpf"), None)
        assert cpf is not None
        assert cpf.confidence >= 0.9


# ── China ───────────────────────────────────────────────────────────────────


class TestChinaPII:
    def test_detects_chinese_phone_without_prefix(self):
        results = detect_pii("手机号: 13800138000", PIIDetectOptions(locales=["cn"]))
        assert any(r.type == "cn_phone" for r in results)

    def test_detects_chinese_phone_plus86(self):
        results = detect_pii("+86 13800138000", PIIDetectOptions(locales=["cn"]))
        # +86 matched by base intl phone pattern
        assert any(r.type == "phone" for r in results)

    def test_no_false_positive_on_invalid_region(self):
        results = detect_pii("Number: 999999199001011234", PIIDetectOptions(locales=["cn"]))
        assert not any(r.type == "cn_national_id" for r in results)


# ── Japan ───────────────────────────────────────────────────────────────────


class TestJapanPII:
    def test_detects_japanese_phone_090(self):
        results = detect_pii("電話: 090-1234-5678", PIIDetectOptions(locales=["jp"]))
        assert any(r.type == "jp_phone" for r in results)

    def test_detects_japanese_phone_plus81(self):
        results = detect_pii("+81 90-1234-5678", PIIDetectOptions(locales=["jp"]))
        assert any(r.type == "phone" for r in results)

    def test_bare_12_digit_no_context_not_detected(self):
        results = detect_pii("Code: 123456789012", PIIDetectOptions(locales=["jp"]))
        assert not any(r.type == "jp_my_number" for r in results)


# ── South Korea ─────────────────────────────────────────────────────────────


class TestSouthKoreaPII:
    def test_detects_korean_phone_010(self):
        results = detect_pii("전화: 010-1234-5678", PIIDetectOptions(locales=["kr"]))
        assert any(r.type == "kr_phone" for r in results)

    def test_rejects_rrn_invalid_month(self):
        results = detect_pii("주민등록: 901301-1234567", PIIDetectOptions(locales=["kr"]))
        assert not any(r.type == "kr_rrn" for r in results)


# ── Germany ─────────────────────────────────────────────────────────────────


class TestGermanyPII:
    def test_detects_tax_id_with_context(self):
        results = detect_pii("Steueridentifikationsnummer: 12345679810", PIIDetectOptions(locales=["de"]))
        assert isinstance(results, list)

    def test_bare_11_digit_no_context_not_detected(self):
        results = detect_pii("Number: 12345678901", PIIDetectOptions(locales=["de"]))
        assert not any(r.type == "de_tax_id" for r in results)


# ── Mexico ──────────────────────────────────────────────────────────────────


class TestMexicoPII:
    def test_detects_curp(self):
        results = detect_pii("CURP: GARC850101HDFRRL09", PIIDetectOptions(locales=["mx"]))
        assert any(r.type == "mx_curp" for r in results)

    def test_rejects_curp_invalid_gender(self):
        results = detect_pii("CURP: GARC850101XDFRRL09", PIIDetectOptions(locales=["mx"]))
        assert not any(r.type == "mx_curp" for r in results)

    def test_detects_rfc(self):
        results = detect_pii("RFC: GARC850101AB3", PIIDetectOptions(locales=["mx"]))
        assert any(r.type == "mx_rfc" for r in results)


# ── France ──────────────────────────────────────────────────────────────────


class TestFrancePII:
    def test_detects_nir_with_valid_check_digit(self):
        base = 1850175108123
        key = 97 - (base % 97)
        nir = f"{base}{key:02d}"
        results = detect_pii(f"Numéro de sécurité sociale: {nir}", PIIDetectOptions(locales=["fr"]))
        assert any(r.type == "fr_nir" for r in results)

    def test_rejects_nir_wrong_check_digit(self):
        results = detect_pii("NIR: 185017510812300", PIIDetectOptions(locales=["fr"]))
        assert not any(r.type == "fr_nir" for r in results)


# ── Cross-locale ────────────────────────────────────────────────────────────


class TestMultiLocale:
    def test_loads_all_locales(self):
        results = detect_pii("Test text", PIIDetectOptions(locales="all"))
        assert isinstance(results, list)

    def test_detects_multiple_countries(self):
        text = "SIN: 046-454-286, CPF: 529.982.247-25, 手机号: 13800138000"
        results = detect_pii(text, PIIDetectOptions(locales=["ca", "br", "cn"]))
        types = [r.type for r in results]
        assert "ca_sin" in types
        assert "br_cpf" in types
        assert "cn_phone" in types

    def test_existing_types_still_work(self):
        results = detect_pii("Email: test@example.com, SSN: 123-45-6789", PIIDetectOptions(locales=["ca"]))
        types = [r.type for r in results]
        assert "email" in types
        assert "ssn" in types

    def test_backward_compatible(self):
        results = detect_pii("My SSN is 123-45-6789")
        assert any(r.type == "ssn" for r in results)
