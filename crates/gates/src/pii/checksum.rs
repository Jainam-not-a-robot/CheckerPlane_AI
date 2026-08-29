//! # Checksum Algorithms for Structured PII Validation
//!
//! **Responsibility:** Implements the Luhn algorithm for payment card verification and the Verhoeff
//! algorithm for 12-digit Indian Aadhaar number validation.
//! **Pipeline Position:** Deterministic check during PII gate evaluation.
//! **Latency Budget:** <1 µs per candidate number.
//! **Failure Mode:** Infallible boolean validators.

/// Validates a credit/debit payment card number using the Luhn checksum algorithm (Mod-10).
#[must_use]
pub fn validate_luhn(card_number: &str) -> bool {
    let digits: Vec<u32> = card_number.chars().filter_map(|c| c.to_digit(10)).collect();

    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }

    let mut sum = 0;
    let mut double = false;

    for &digit in digits.iter().rev() {
        if double {
            let doubled = digit * 2;
            sum += if doubled > 9 { doubled - 9 } else { doubled };
        } else {
            sum += digit;
        }
        double = !double;
    }

    sum % 10 == 0
}

/// Verhoeff algorithm multiplication table d(j, k).
const VERHOEFF_D: [[usize; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

/// Verhoeff algorithm permutation table p(pos % 8, num).
const VERHOEFF_P: [[usize; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

/// Validates a 12-digit Indian Aadhaar number using the Verhoeff checksum algorithm.
#[must_use]
pub fn validate_verhoeff(aadhaar_number: &str) -> bool {
    let digits: Vec<usize> = aadhaar_number
        .chars()
        .filter_map(|c| c.to_digit(10).map(|d| d as usize))
        .collect();

    if digits.len() != 12 {
        return false;
    }

    // Aadhaar cannot begin with 0 or 1
    if digits[0] < 2 {
        return false;
    }

    let mut c = 0;
    for (i, &digit) in digits.iter().rev().enumerate() {
        let p_val = VERHOEFF_P[i % 8][digit];
        c = VERHOEFF_D[c][p_val];
    }

    c == 0
}
