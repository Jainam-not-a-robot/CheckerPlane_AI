//! # Deterministic PII Regex Pattern Matching
//!
//! **Responsibility:** Implements high-precision regex matching combined with mathematical checksums
//! for detecting high-risk identifiers (Cards, Aadhaar, PAN, Credentials) and observation entities.
//! **Pipeline Position:** Deterministic first-pass inside the PII gate.
//! **Latency Budget:** <500 µs.
//! **Failure Mode:** Infallible pattern matcher.

use crate::pii::checksum::{validate_luhn, validate_verhoeff};
use regex::Regex;
use std::sync::LazyLock;

static PAYMENT_CARD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11}|[0-9]{4}[ -][0-9]{4}[ -][0-9]{4}[ -][0-9]{4})\b")
        .expect("valid regex")
});

static AADHAAR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[2-9]{1}[0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}\b").expect("valid regex")
});

static PAN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b").expect("valid regex"));

static JWT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\beyJ[A-Za-z0-9-_]{10,}\.[A-Za-z0-9-_]{10,}\.[A-Za-z0-9-_]{10,}\b")
        .expect("valid regex")
});

static PRIVATE_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----").expect("valid regex")
});

static API_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|bearer)\s*[:=]\s*['"]?([A-Za-z0-9_\-\.]{16,})['"]?"#)
        .expect("valid regex")
});

static EMAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").expect("valid regex")
});

static PHONE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:\+?[1-9]\d{0,2}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")
        .expect("valid regex")
});

/// High-risk PII pattern match finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternFinding {
    /// Identifier category class (e.g. "payment_card", "aadhaar", "pan", "credential").
    pub class_name: &'static str,
    /// Masked snippet or match summary.
    pub matched_value: String,
    /// Character offset in text.
    pub start: usize,
    pub end: usize,
}

/// Scans text for high-risk structured PII patterns with mathematical checksum verification.
#[must_use]
pub fn scan_high_risk_patterns(text: &str) -> Vec<PatternFinding> {
    let mut findings = Vec::new();

    // 1. Payment Cards (Regex + Luhn)
    for mat in PAYMENT_CARD_RE.find_iter(text) {
        let raw = mat.as_str();
        if validate_luhn(raw) {
            findings.push(PatternFinding {
                class_name: "payment_card",
                matched_value: mask_identifier(raw),
                start: mat.start(),
                end: mat.end(),
            });
        }
    }

    // 2. Indian Aadhaar Number (Regex + Verhoeff)
    for mat in AADHAAR_RE.find_iter(text) {
        let raw = mat.as_str();
        if validate_verhoeff(raw) {
            findings.push(PatternFinding {
                class_name: "aadhaar",
                matched_value: mask_identifier(raw),
                start: mat.start(),
                end: mat.end(),
            });
        }
    }

    // 3. Indian PAN Card (Strict Regex)
    for mat in PAN_RE.find_iter(text) {
        findings.push(PatternFinding {
            class_name: "pan",
            matched_value: mask_identifier(mat.as_str()),
            start: mat.start(),
            end: mat.end(),
        });
    }

    // 4. Credentials: JWT Tokens
    for mat in JWT_RE.find_iter(text) {
        findings.push(PatternFinding {
            class_name: "credential",
            matched_value: "JWT_TOKEN[redacted]".to_string(),
            start: mat.start(),
            end: mat.end(),
        });
    }

    // 5. Credentials: Private Keys
    for mat in PRIVATE_KEY_RE.find_iter(text) {
        findings.push(PatternFinding {
            class_name: "credential",
            matched_value: "PRIVATE_KEY_BLOCK[redacted]".to_string(),
            start: mat.start(),
            end: mat.end(),
        });
    }

    // 6. Credentials: API Keys
    for mat in API_KEY_RE.find_iter(text) {
        findings.push(PatternFinding {
            class_name: "credential",
            matched_value: "API_SECRET_KEY[redacted]".to_string(),
            start: mat.start(),
            end: mat.end(),
        });
    }

    findings
}

/// Scans text for observation-level entities (emails, phones).
#[must_use]
pub fn scan_observation_patterns(text: &str) -> Vec<PatternFinding> {
    let mut findings = Vec::new();

    for mat in EMAIL_RE.find_iter(text) {
        findings.push(PatternFinding {
            class_name: "email",
            matched_value: mask_identifier(mat.as_str()),
            start: mat.start(),
            end: mat.end(),
        });
    }

    for mat in PHONE_RE.find_iter(text) {
        findings.push(PatternFinding {
            class_name: "phone",
            matched_value: mask_identifier(mat.as_str()),
            start: mat.start(),
            end: mat.end(),
        });
    }

    findings
}

/// Masks sensitive identifier values for safe logging in response details.
fn mask_identifier(val: &str) -> String {
    let len = val.len();
    if len <= 4 {
        return "****".to_string();
    }
    let suffix = &val[len - 4..];
    format!("****-{}", suffix)
}
