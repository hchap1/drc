# pygame_viewer.py

import io
import queue
import socket
import struct
import threading
import time

import pygame

from client import connect

JETSON_IP = '192.168.4.1'

def main() -> None:
    c = connect(JETSON_IP)
    pygame.init()
    screen = pygame.display.set_mode((640, 480), pygame.RESIZABLE)
    pygame.display.set_caption('Jetson feed')
    clock = pygame.time.Clock()
    font = pygame.font.Font('freesansbold.ttf', 25)

    speed = 0.1
    running = True

    while running:
        dt = clock.tick(100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        key = pygame.key.get_pressed()

        left = 0.0
        right = 0.0

        if key[pygame.K_w]:
            left += speed
            right += speed
        if key[pygame.K_d]:
            left += speed * 1.5
            right -= speed * 1.5
        if key[pygame.K_a]:
            left -= speed * 1.5
            right += speed * 1.5
        if key[pygame.K_s]:
            left -= speed
            right -= speed

        if key[pygame.K_SPACE]:
            speed += dt / 10000
        if key[pygame.K_LSHIFT]:
            speed -= dt / 10000

        speed = max(0.05, min(0.9, speed))

        screen.fill((255, 255, 255))
        screen.blit(font.render(f"SPEED: {speed}", True, (0, 0, 0)), (10, 10))
        pygame.display.flip()

        c.send(left, right)

    c.close()
    pygame.quit()


if __name__ == '__main__':
    main()

