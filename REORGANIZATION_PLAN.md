# Project Reorganization Plan

## Current Issues
1. Documentation files scattered in root directory
2. Test outputs committed to repository
3. __pycache__ directories in version control
4. No clear separation between user docs and developer docs

## Proposed New Structure

```
Ear-segmentation-ai/
│
├── .github/                    # GitHub specific files (workflows, templates)
│   ├── workflows/             # CI/CD workflows
│   └── ISSUE_TEMPLATE/        # Issue templates
│
├── assets/                     # Project assets
│   ├── images/                # Images for documentation
│   └── logo/                  # Project logo/branding
│
├── docs/                       # All documentation
│   ├── api/                   # API documentation (auto-generated)
│   ├── guides/                # User guides and tutorials
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── advanced.md
│   ├── development/           # Developer documentation
│   │   ├── CLAUDE.md
│   │   ├── CONTRIBUTING.md
│   │   └── architecture.md
│   ├── project/               # Project governance
│   │   ├── CODE_OF_CONDUCT.md
│   │   ├── SECURITY.md
│   │   └── CHANGELOG.md
│   └── migration/             # Migration guides
│       └── MIGRATION.md
│
├── examples/                   # Example code and notebooks
│   ├── basic/                 # Basic usage examples
│   ├── advanced/              # Advanced examples
│   └── notebooks/             # Jupyter notebooks
│
├── scripts/                    # Development and utility scripts
│   ├── development/           # Development tools
│   ├── release/               # Release automation
│   └── benchmarks/            # Performance benchmarks
│
├── src/                        # Source code (no changes)
│   └── earsegmentationai/
│
├── tests/                      # Test suite (no changes)
│   ├── unit/
│   └── integration/
│
├── .gitignore                 # Updated with new patterns
├── .pre-commit-config.yaml    # Pre-commit hooks
├── LICENSE                    # License file
├── Makefile                   # Development tasks
├── README.md                  # Main readme
├── poetry.lock                # Dependency lock
└── pyproject.toml            # Project configuration
```

## Migration Steps

### Step 1: Create New Directory Structure
```bash
mkdir -p docs/{api,guides,development,project,migration}
mkdir -p assets/{images,logo}
mkdir -p examples/{basic,advanced}
mkdir -p scripts/{development,release,benchmarks}
```

### Step 2: Move Documentation Files
- `CODE_OF_CONDUCT.md` → `docs/project/`
- `CONTRIBUTING.md` → `docs/development/`
- `SECURITY.md` → `docs/project/`
- `CHANGELOG.md` → `docs/project/`
- `MIGRATION.md` → `docs/migration/`
- `CLAUDE.md` → `docs/development/`

### Step 3: Move Scripts
- `scripts/prepare_release.py` → `scripts/release/`
- `scripts/test_*.py` → `scripts/development/`

### Step 4: Clean Up
- Add `test_output/` to `.gitignore`
- Add `**/__pycache__/` to `.gitignore`
- Remove all `__pycache__` directories
- Remove `test_output/` directory

### Step 5: Create New Documentation
- `docs/guides/installation.md` - Extract from README
- `docs/guides/quickstart.md` - Extract from README
- `docs/guides/advanced.md` - Advanced usage guide
- `docs/development/architecture.md` - System architecture

### Step 6: Update References
- Update README.md links to new doc locations
- Update pyproject.toml documentation URL
- Update any import statements if needed

## Benefits
1. **Cleaner root directory** - Only essential files in root
2. **Better organization** - Clear separation of concerns
3. **Easier navigation** - Logical grouping of related files
4. **Professional appearance** - Standard open source project layout
5. **Better documentation** - Separated user and developer docs
