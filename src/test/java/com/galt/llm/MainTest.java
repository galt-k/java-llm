package com.galt.llm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

class MainTest {
    @Test
    void mainExecutesCleanly() {
        assertDoesNotThrow(() -> Main.main(new String[0]));
    }
}
