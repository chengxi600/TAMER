import os
import pygame


class Interface:
    """ Pygame interface for rendering gym environment and training TAMER+RL """

    def __init__(self, action_map, env_frame_shape):
        self.action_map = action_map
        self.env_frame_shape = env_frame_shape
        self.PANEL_HEIGHT = 100
        self.panel_color = (0, 0, 0)

        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        # extend the screen size to display agent action for human feedback
        height, width, _ = self.env_frame_shape
        screen_size = (width, height+100)
        self.screen = pygame.display.set_mode(screen_size)
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)

    def get_scalar_feedback(self):
        """ Gets scalar feedback from key press events. Updates panel color.

        Returns:
            int: scalar reward
        """
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.panel_color = (0, 255, 0)
                    reward = 1
                    self.panel_timer = 0   # reset panel display timer
                elif event.key == pygame.K_a:
                    self.panel_color = (255, 0, 0)
                    reward = -1
                    self.panel_timer = 0
        return reward

    def render(self, env_frame, action):
        """ Renders a frame given environment frame and action. Displays action

        Args:
            env_frame ((int, int, int)): environment frame of shape (x, y, 3)
            action (int): action agent is executing
        """
        surf = pygame.surfarray.make_surface(env_frame.swapaxes(0, 1))

        self.screen.fill((0, 0, 0))
        self.screen.blit(surf, (0, 0))

        # start drawing right below the gym env
        env_height, env_width, _ = self.env_frame_shape
        panel_rect = pygame.Rect(
            0, env_height, env_width, self.PANEL_HEIGHT)
        pygame.draw.rect(self.screen, self.panel_color,
                         panel_rect)  # background

        # render text
        text = self.font.render(
            self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect(
            center=(env_width // 2, env_height + 50))
        self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.panel_color = (0, 0, 0)

    def close(self):
        pygame.display.quit()
        pygame.quit()
