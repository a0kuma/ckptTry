# want know

around line 823 of ckpt py

talk about class checkpoint frame

inside the class have an attr : weak holder

that is a list of weak ref type

that is you put weak ref inside ?

## if we go from top to bottom

1. checkpoint frame 
2. list weak holders is list of CLASS holders
3. created at _checkpoint_hook append stuff arround line 1162

1. that is:
2. create a empty object Holder and put in weak holder list
