def play_hangman(model, masked_word, true_word, char_to_idx, idx_to_char):
    import numpy as np

    guessed_letters = set()
    wrong_guesses = 0
    max_wrong_guesses = 6

    print("Starting Hangman Game")
    print("True word (for debugging):", true_word)

    while '_' in masked_word and wrong_guesses < max_wrong_guesses:
        print("\nCurrent Word:", ''.join(masked_word))
        input_seq = [char_to_idx[c] for c in masked_word]
        padded_seq = input_seq + [char_to_idx['_']] * (20 - len(input_seq))  # Pad to MAX_LEN

        # Reshape and predict
        pred = model.predict(np.array([padded_seq]), verbose=0)[0]
        sorted_indices = np.argsort(pred)[::-1]

        next_guess = None
        for idx in sorted_indices:
            guess = idx_to_char[idx]
            if guess not in guessed_letters and guess != '_':
                next_guess = guess
                break

        guessed_letters.add(next_guess)
        print(f"Model guessed: {next_guess}")

        if next_guess in true_word:
            # Replace blanks in masked_word where guess is correct
            for i, char in enumerate(true_word):
                if char == next_guess:
                    masked_word[i] = next_guess
            print("Correct!")
        else:
            wrong_guesses += 1
            print(f"Wrong guess! Remaining tries: {max_wrong_guesses - wrong_guesses}")

    print("\nFinal Word:", ''.join(masked_word))
    if '_' not in masked_word:
        print("🎉 Model WON!")
    else:
        print("💀 Model LOST!")
        
# Example usage:
# play_hangman(model_name,list(true word string),'true word',char_to_idx,idx_to_char)