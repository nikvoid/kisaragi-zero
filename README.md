# Kisaragi Zero
My multi-purpose discord bot.
Expected to be used on one or few specified servers (guilds).

## Features
 - `/dream` and `/dream_matrix` commands - communicate with stable diffusion backend
   (such as `stable-diffusion-webui` or `naifu`) to generate images
 - `/roll` - roll one of predefined dices
 - `/hello` - greet user, test command
 - `$$emojis` and `$$<emoji_name>` - send arbitrary images from `emoji` folder

## Config
See example configuration in file `example-config.toml`.
Bot accepts path to config as first command line argument, otherwise it defaults to `config.toml`.

## Resources
 - `hanyuu.png` - used by mock SD backend
 - `nunito.ttf` - used to print generation parameters on `/dream_matrix` output