#!/usr/bin/env bash
# Check that all team slugs in CODEOWNERS exist in the organization
set -euo pipefail

ORG=EffortlessMetrics

# Fetch all team slugs in the org
echo "Fetching teams from @$ORG..."
mapfile -t SLUGS < <(gh api "orgs/$ORG/teams?per_page=100" --paginate | jq -r '.[].slug' | sort -u)

if [ ${#SLUGS[@]} -eq 0 ]; then
  echo "ERROR: Failed to fetch team slugs (check gh auth and permissions)"
  exit 1
fi

echo "Found ${#SLUGS[@]} teams in @$ORG"
echo

ok=0
bad=0

# Extract @org/team tokens from CODEOWNERS
if [ ! -f CODEOWNERS ]; then
  echo "ERROR: CODEOWNERS file not found"
  exit 1
fi

mapfile -t TEAMS < <(grep -o '@[^/[:space:]]\+/[^[:space:]]\+' CODEOWNERS \
                     | sed -E 's/^@//; s/@.*$//' \
                     | sort -u)

if [ ${#TEAMS[@]} -eq 0 ]; then
  echo "No team references found in CODEOWNERS"
  exit 0
fi

echo "Validating ${#TEAMS[@]} team reference(s) from CODEOWNERS:"
echo

for t in "${TEAMS[@]}"; do
  org="${t%%/*}"
  slug="${t#*/}"

  if [[ "$org" != "$ORG" ]]; then
    echo "⚠️  WARN: @$t (different org)"
    continue
  fi

  if printf '%s\n' "${SLUGS[@]}" | grep -qx "$slug"; then
    echo "✅ OK: @$ORG/$slug"
    ((ok++))
  else
    echo "❌ BAD: @$ORG/$slug (team not found in organization)"
    ((bad++))
  fi
done

echo
echo "Summary: ✅ $ok OK, ❌ $bad BAD"

if [ $bad -gt 0 ]; then
  echo
  echo "Fix these issues by:"
  echo "  1. Creating the missing teams in GitHub"
  echo "  2. Updating CODEOWNERS with correct team slugs"
  exit 1
fi

echo
echo "All CODEOWNERS team slugs are valid!"
