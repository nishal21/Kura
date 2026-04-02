use keyring::Entry;
use std::fmt::{Display, Formatter};

const APP_SERVICE_PREFIX: &str = "kura";

#[derive(Debug)]
pub enum SecureStoreError {
    InvalidProvider,
    EmptyApiKey,
    CredentialStorePermissionDenied { provider: String, details: String },
    CredentialStoreFailure { provider: String, details: String },
}

impl Display for SecureStoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProvider => {
                write!(f, "provider is required")
            }
            Self::EmptyApiKey => {
                write!(f, "api_key is required")
            }
            Self::CredentialStorePermissionDenied { provider, details } => {
                write!(
                    f,
                    "Permission denied while accessing secure credential store for provider '{}': {}",
                    provider, details
                )
            }
            Self::CredentialStoreFailure { provider, details } => {
                write!(
                    f,
                    "Failed to access secure credential store for provider '{}': {}",
                    provider, details
                )
            }
        }
    }
}

impl std::error::Error for SecureStoreError {}

pub fn save_api_key_secure(provider: &str, api_key: &str) -> Result<(), SecureStoreError> {
    let normalized_provider = normalize_provider(provider)?;
    let normalized_key = normalize_api_key(api_key)?;

    let entry = entry_for_provider(&normalized_provider)?;

    entry
        .set_password(normalized_key)
        .map_err(|error| map_keyring_error(&normalized_provider, error))
}

pub fn check_api_key_secure(provider: &str) -> Result<bool, SecureStoreError> {
    let normalized_provider = normalize_provider(provider)?;
    let entry = entry_for_provider(&normalized_provider)?;

    match entry.get_password() {
        Ok(saved_key) => Ok(!saved_key.trim().is_empty()),
        Err(keyring::Error::NoEntry) => Ok(false),
        Err(error) => Err(map_keyring_error(&normalized_provider, error)),
    }
}

fn normalize_provider(provider: &str) -> Result<String, SecureStoreError> {
    let trimmed = provider.trim();
    if trimmed.is_empty() {
        return Err(SecureStoreError::InvalidProvider);
    }

    Ok(trimmed.to_ascii_lowercase().replace(' ', "_"))
}

fn normalize_api_key(api_key: &str) -> Result<&str, SecureStoreError> {
    let trimmed = api_key.trim();
    if trimmed.is_empty() {
        return Err(SecureStoreError::EmptyApiKey);
    }

    Ok(trimmed)
}

fn entry_for_provider(provider: &str) -> Result<Entry, SecureStoreError> {
    let service = format!("{}.{}", APP_SERVICE_PREFIX, provider);

    Entry::new(&service, provider)
        .map_err(|error| map_keyring_error(provider, error))
}

fn map_keyring_error(provider: &str, error: keyring::Error) -> SecureStoreError {
    let details = error.to_string();

    if is_permission_denied_message(&details) {
        return SecureStoreError::CredentialStorePermissionDenied {
            provider: provider.to_string(),
            details,
        };
    }

    SecureStoreError::CredentialStoreFailure {
        provider: provider.to_string(),
        details,
    }
}

fn is_permission_denied_message(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();

    normalized.contains("permission denied")
        || normalized.contains("access is denied")
        || normalized.contains("not permitted")
        || normalized.contains("operation not permitted")
        || normalized.contains("not allowed")
        || normalized.contains("authorization")
        || normalized.contains("unauthorized")
}
