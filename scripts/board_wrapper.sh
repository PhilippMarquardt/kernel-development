#!/bin/bash
# Legacy `board` interface (interim /root/okc/board/board.sh) on top of `kdev board` (docs/COLLAB.md).
# Install: cp scripts/board_wrapper.sh /usr/local/bin/board && chmod +x /usr/local/bin/board
#   board post <agent> <kind> <message...>   kind: FINDING | CLAIM | DONE | ASK | ANSWER | WARN | DECISION
#   board read [N]        last N posts (default 40)
#   board since <HH:MM>   posts since a UTC time today
#   board grep <pattern>
#   board inbox <agent>   posts addressed to / mentioning <agent>
# Anything else is passed through: `board claims`, `board thread 12`, ... == `kdev board ...`.
KDEV=${KDEV:-kdev}
case "$1" in
  post)  a=$2; k=$(echo "$3" | tr '[:upper:]' '[:lower:]'); shift 3
         exec "$KDEV" board post "$k" "$*" --agent "$a" ;;
  read)  exec "$KDEV" board read -n "${2:-40}" ;;
  since) exec "$KDEV" board read --since "$2" -n 200 ;;
  grep)  exec "$KDEV" board search "$2" -n 100 ;;
  inbox) exec "$KDEV" board inbox --agent "$2" ;;
  ""|-h|--help|help) sed -n 2,9p "$0" ;;
  *)     exec "$KDEV" board "$@" ;;
esac
