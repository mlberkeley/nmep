---
title: Using Git
parent: Resources
nav_order: 2
layout: home
---

# Using Git

GitHub is an essential tool used in the ML community and beyond to host codebases, collaborate on projects, deploy websites (like this one!) and much more. We'll also be using it for many of your NMEP homeworks. Below we'll list some helpful information/setup for anybody familiarizing themsleves with Git, the version control system behind GitHub.

Before getting started, make an account on [GitHub](https://github.com/), if you haven't already!

## SSH Setup

SSH (which stands for Secure Shell Protocol) allows users to authenticate with GitHub servers without using a username and password. Instead, by using SSH keys specific to each computer, it's possible to use GitHub's services, like cloning repositories, committing changes, and more, nearly automatically. 

If you haven't already, create a new SSH key for your computer using [these directions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent), making sure to choose your operating system. If you think you already have an SSH key, you can check [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys).

Now that you have an SSH key, we need to register this key with GitHub so that their servers can recognize your computer and give you the appropriate access. Usually, you can find your public key at `~/.ssh/id_ed25519.pub`, but this can depend on how you created the SSH key and your operating system. For more specific information, check [this site](docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account). Now, in GitHub settings, you should be able to paste this public key to register your device. 

After you complete these steps, you should be able to clone repositories using the recommended SSH link.

## Extra Information

- Berkeley's very own CS 61B has excellent documentation behind Git and its features. We highly recommend taking a look at it [here](https://sp25.datastructur.es/resources/guides/git/). 