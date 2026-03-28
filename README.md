# what im building

in-game assistant for my favorite mobile game: brawl stars :D

# why

- saw how chess recently expanded their ai team/hired GM Vinay Bhat as a director to work towards developing a coach which, "knows what you've been doing, what are some of your
strengths, weaknesses, patterns. It knows how to encourage you, sort of lean into positive reinforcement and some of the behaviors, habits."

- ai is meant to be a coach that's able to continuously open up a complex game like chess to more and more players

- why not apply this same philosophy to a dynamic mobile game which i happen to be pretty good at! especially considering that the game has gotten significantly more
complicated since i started 6+ years ago and i know many people who started recently and are struggling to figure out what to do during specific parts of the game! 10M+
active players and there's very few free resources online to improve 

# curr progress

used models to identify characters on loading screen with high accuracy using images from the wiki (there's some issues with identifying brawlers with skins that 
significantly change the model but minor)

connects directly to your phone to broadcast the game & detect loading screen from live gameplay

uses gemini's live api to give real time insights into how to play into those speciifc brawlers + the mode
