#!/bin/bash

chmod +x "$PWD"/scripts/pre-push
ln -s "$PWD"/scripts/pre-push "$PWD"/.git/hooks/pre-push

GREEN=$'\e[0;32m';
NC=$'\e[0m'
echo "${GREEN}Hooks installed${NC}"
