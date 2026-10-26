# Quick answer (SigLIP2 search only): how good is its count?

Claude-labelled, human audit pending; same 164 people x 8 questions as `results.md`.

- SigLIP2's own probability, sigmoid(112.67 * cos - 16.77), is far too low on these crops: summed over people it gives
  0.1 to 4.0 against true counts of 9 to 25 (mean error 90 %). Unusable as a count.
- Rule used in the demo: match = similarity more than z = 1.25 standard deviations above the video's mean.
  Mean absolute count error 26 % (leave-one-question-out, so the z choice is not tuned on the question scored).
- Caveat: that rule predicts 11 to 15 for every question, and my 8 questions all have 9 to 25 true matches. It is
  mostly a prior ("about a tenth of people match"), not signal. For a rare attribute (true count 1) it would say ~13.
  The demo therefore shows the ranking and thumbnails as the real answer and the number as a rough guide.
- Ranking quality (top-k hits where k = true count): 6/10 red top, 15/20 white shirt, 6/12 black jacket, 5/9 backpack,
  12/25 bag, 10/16 shorts, 6/10 hat, 2/11 shopping bag.
