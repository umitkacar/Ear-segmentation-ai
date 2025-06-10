# Security Policy

## 🔒 Security is Our Priority

We take the security of Ear Segmentation AI seriously. This document outlines our security policy and procedures for reporting vulnerabilities.

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within Ear Segmentation AI, please follow these steps:

### 📧 Contact

Send an email to the maintainers at:
- **Primary**: umitkacar.phd@gmail.com
- **Secondary**: thunderbirdtr@gmail.com

### 📝 What to Include

Please include the following information:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential impact and severity assessment
3. **Steps to Reproduce**: Detailed steps to reproduce the issue
4. **Proof of Concept**: Example code or demonstration (if applicable)
5. **Suggested Fix**: Your recommendations for fixing (if any)
6. **Environment**: Python version, OS, and package versions

### ⏱️ Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Resolution Target**: Within 30 days for critical issues

## Security Measures

### 🛡️ Current Security Features

- Input validation for all image/video processing
- Safe file handling with path validation
- Dependency security scanning via GitHub Dependabot
- Regular security updates for dependencies
- Safe model loading with checksum verification

### 🔍 Security Best Practices

When using Ear Segmentation AI:

1. **Model Files**: Only load models from trusted sources
2. **Input Validation**: Always validate user inputs in production
3. **File Permissions**: Use appropriate file permissions for outputs
4. **Dependencies**: Keep dependencies up to date
5. **Network**: Use HTTPS for model downloads

## Disclosure Policy

- **Private Disclosure**: Please do not publicly disclose vulnerabilities
- **Coordinated Disclosure**: We'll work with you on disclosure timing
- **Credit**: Security researchers will be credited (unless anonymity is requested)
- **CVE**: We'll request CVEs for significant vulnerabilities

## Security Updates

Security updates are released as:
- **Patch versions** for non-breaking security fixes
- **Minor versions** for security fixes requiring small API changes
- **Security advisories** via GitHub Security Advisories

## Out of Scope

The following are not considered vulnerabilities:

- Issues in dependencies (report to the dependency maintainer)
- Issues requiring physical access to the system
- Social engineering attacks
- Denial of service attacks that don't exploit a specific vulnerability

## Recognition

We appreciate the security research community's efforts in helping keep Ear Segmentation AI secure. Researchers who report valid security issues will be acknowledged in our release notes and security advisories.

---

Thank you for helping us keep Ear Segmentation AI secure! 🙏
